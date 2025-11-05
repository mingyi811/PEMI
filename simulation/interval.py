import numpy as np
import math
from typing import Tuple, List, Callable
from config import j_feature, method, reference_set_method
from selection import selection_rule_covariate_dependent
from quantile_forest import RandomForestQuantileRegressor
from typing import Optional

def get_smoothed_cutoff_with_N1(scores: List[float], alpha: float, N1: int, quantile_method: str) -> float:
    """
    Compute p-value for a given v_test using randomized smoothing.
    p(y) = (#{scores > v_test} + U * (#{scores == v_test} + N1)) / (len(scores) + N1)
    """
    if quantile_method == "upper":
        scores_arr = np.array(scores) 
        total = len(scores_arr) + N1 
        v_upper = math.inf
        for v in np.sort(np.unique(scores_arr))[::-1]:
            num_gt = np.sum(scores_arr > v)
            num_eq = np.sum(scores_arr == v)
            p_val = (num_gt + (num_eq + N1)) / (total)
            if p_val > alpha: 
                return v_upper 
            v_upper = v
        return v_upper

    elif quantile_method == "randomize":
        scores_arr = np.array(scores)
        total = len(scores_arr) + N1
        U=np.random.uniform(0,1)
        v_upper = math.inf
        for v in np.sort(np.unique(scores_arr))[::-1]:
            num_gt = np.sum(scores_arr > v)
            num_eq = np.sum(scores_arr == v)
            p_val = (num_gt + U * (num_eq + N1)) / (total)
            if p_val > alpha:
                return v_upper
            v_upper = v
        return v_upper
    else:
        raise ValueError(f"Unknown method: {quantile_method!r}")




def construct_prediction_interval(
    t: int,
    Y_on: np.ndarray,
    mu_on: np.ndarray,
    Y_off: np.ndarray,
    mu_off: np.ndarray,
    X_off: np.ndarray,
    X_on: np.ndarray,
    M: int,
    alpha: float,
    quantile_method: str = "upper",
    reference_set_method: str = reference_set_method,
    qrf_model: Optional[RandomForestQuantileRegressor] = None
) -> Tuple[float, float, int]:
    """
    Build symmetric prediction interval at time t.
    This function works with selection_rule of form (X_j_val, cum_selected) -> bool
    or (mu_t, mu_history) -> bool, depending on the rule logic.
    """
    if reference_set_method == "ours":
        n_off = len(mu_off)
        X_aug = np.concatenate((X_off,X_on))
        mu_aug = np.concatenate((mu_off,mu_on))
        Y_aug = np.concatenate((Y_off,Y_on))
        if t+n_off == 0:
            return -math.inf, math.inf, 0
        i = t
        indices_aug = list(range(i + n_off + 1))
        base_perm_aug = tuple(indices_aug)  # base_perm is the permutation of observed data

        perms_list_aug = [base_perm_aug]
        max_perms_aug = min(M + 1, math.factorial(i + 1 + n_off))
        # perms_list is the list of initial random permutations

        while len(perms_list_aug) < max_perms_aug:
            pi = tuple(np.random.permutation(indices_aug))
            if pi not in perms_list_aug:
                perms_list_aug.append(pi)

        perms_selected = []
        for pi in perms_list_aug:
            mu_hist = mu_aug[list(pi)][n_off:i+n_off]  # From n_off to i-1, exclude offline data
            mu_test = mu_aug[list(pi)][i+n_off]
            if selection_rule_covariate_dependent(mu_test, mu_hist,method=method):
                perms_selected.append(pi)
        cal_size = len(perms_selected)

        # score function: residual
        # N1 = 0
        # scores: List[float] = []
        # for pi in perms_selected:
        #     if pi[i+n_off] == i+n_off:
        #         N1 += 1  # N1: we don't know exact score but know this permutation's score equals observed data's score
        #     else:
        #         k = pi[i+n_off]
        #         scores.append(compute_cqr_score(qrf_model, X_aug[k], Y_aug[k]))
        #         # scores.append(abs(Y_aug[k] - mu_aug[k]))  # Use residual of the last point as score

        # if not scores:
        #     return -math.inf, math.inf, 0
        # scores.append(math.inf)  # Append inf to scores

        # score function: CQR score
        N1 = 0
        indices = []  # Collect indices where pi[i+n_off] != i+n_off
        for pi in perms_selected:
            if pi[i+n_off] == i+n_off:
                N1 += 1  # Count permutations where score equals observed data's score
            else:
                k = pi[i+n_off]
                indices.append(k)  # Collect index k for batch processing

        if not indices:
            return -math.inf, math.inf, 0

        # Compute scores for all collected indices in a batch
        X_batch = X_aug[indices]  # Shape: (n_indices, n_features)
        Y_batch = Y_aug[indices]  # Shape: (n_indices,)
        scores = compute_cqr_score_batch(qrf_model, X_batch, Y_batch).tolist()
        scores.append(math.inf)  # Append inf as before

        v_cut = get_smoothed_cutoff_with_N1(scores, alpha, N1, quantile_method=quantile_method)
        pred_test = qrf_model.predict(X_aug[n_off+i].reshape(1,-1), quantiles=[0.2, 0.8])
        q_lower = pred_test[:,0][0]
        q_upper = pred_test[:,1][0]

        return q_lower - v_cut, q_upper + v_cut, cal_size


    elif reference_set_method == "jomi":
        n_off = len(mu_off)
        mu_aug = np.concatenate((mu_off,mu_on))
        X_aug = np.concatenate((X_off,X_on))
        Y_aug = np.concatenate((Y_off,Y_on))
        if t+n_off == 0:
            return -math.inf, math.inf, 0
        i = t
        indices_aug = list(range(i + n_off + 1))
        base_perm_aug = tuple(indices_aug)  # base_perm is the permutation of observed data

        perms_list_aug = [base_perm_aug]

        for j in range(i+n_off):
            # Set pi to permutation with positions j and i+n_off swapped
            pi = list(base_perm_aug)
            pi[j], pi[i+n_off] = pi[i+n_off], pi[j]
            perms_list_aug.append(tuple(pi))

        perms_selected = []
        for pi in perms_list_aug:
            mu_hist = mu_aug[list(pi)][n_off:i+n_off]  # From n_off to i-1, exclude offline data
            mu_test = mu_aug[list(pi)][i+n_off]
            if selection_rule_covariate_dependent(mu_test, mu_hist,method=method):
                perms_selected.append(pi)
        cal_size = len(perms_selected)

        # score function: residual
        # N1 = 0
        # scores: List[float] = []
        # for pi in perms_selected:
        #     if pi[i+n_off] == i+n_off:
        #         N1 += 1  # N1: we don't know exact score but know this permutation's score equals observed data's score
        #     else:
        #         k = pi[i+n_off]

        #         scores.append(compute_cqr_score(qrf_model, X_aug[k], Y_aug[k]))
        #         # scores.append(abs(Y_aug[k] - mu_aug[k]))  # Use residual of the last point as score

        # if not scores:
        #     return -math.inf, math.inf, 0
        # scores.append(math.inf)  # Append inf to scores
        
        # score function: CQR
        N1 = 0
        indices = []  # Collect indices where pi[i+n_off] != i+n_off
        for pi in perms_selected:
            if pi[i+n_off] == i+n_off:
                N1 += 1  # Count permutations where score equals observed data's score
            else:
                k = pi[i+n_off]
                indices.append(k)  # Collect index k for batch processing

        if not perms_selected:
            return -math.inf, math.inf, 0

        # Compute scores for all collected indices in a batch
        X_batch = X_aug[indices]  # Shape: (n_indices, n_features)
        Y_batch = Y_aug[indices]  # Shape: (n_indices,)
        scores = compute_cqr_score_batch(qrf_model, X_batch, Y_batch).tolist()
        scores.append(math.inf)  # Append inf as before

        v_cut = get_smoothed_cutoff_with_N1(scores, alpha, N1, quantile_method=quantile_method)
        pred_test = qrf_model.predict(X_aug[n_off+i].reshape(1,-1), quantiles=[0.2, 0.8])
        q_lower = pred_test[:,0][0]
        q_upper = pred_test[:,1][0]

        return q_lower - v_cut, q_upper + v_cut, cal_size


    elif reference_set_method == "vanilla":
        n_off = len(mu_off)
        mu_aug = np.concatenate((mu_off,mu_on))
        X_aug = np.concatenate((X_off,X_on))
        Y_aug = np.concatenate((Y_off,Y_on))
        if t+n_off == 0:
            return -math.inf, math.inf, 0
        i = t
        N1 = 0
        cal_size = i+n_off

        scores: List[float] = []

        # score function: residual
        # scores=list(np.abs(Y_aug[n_off:n_off+i]-mu_aug[n_off:n_off+i]))
        # scores.append(math.inf)  # Append inf to scores

        # v_cut = np.quantile(scores, 1-alpha, interpolation='higher')
        # return mu_aug[n_off+i] - v_cut, mu_aug[n_off+i] + v_cut, cal_size

        # score function: CQR
        scores = compute_cqr_score_batch(qrf_model, X_aug[n_off:n_off+i], Y_aug[n_off:n_off+i]).tolist()
        scores.append(math.inf)  # Append inf as before
        

        pred_test = qrf_model.predict(X_aug[n_off+i].reshape(1,-1), quantiles=[0.2, 0.8])
        q_lower = pred_test[:,0][0]
        q_upper = pred_test[:,1][0]

        v_cut = get_smoothed_cutoff_with_N1(scores, alpha, N1, quantile_method=quantile_method)
        return q_lower - v_cut, q_upper + v_cut, cal_size
        

def compute_cqr_score(qrf_model: RandomForestQuantileRegressor, X: np.ndarray, Y: float) -> float:
    """
    Compute CQR score for a given X and Y.
    """
    pred = qrf_model.predict(X.reshape(1,-1), quantiles=[0.2, 0.8])
    q_lower = pred[:,0][0]
    q_upper = pred[:,1][0]
    score=max(q_lower-Y, Y-q_upper)
    return score

def compute_cqr_score_batch(qrf_model, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute CQR scores for a batch of X and Y.
    
    Args:
        qrf_model: RandomForestQuantileRegressor model
        X: np.ndarray of shape (n_samples, n_features), input features
        Y: np.ndarray of shape (n_samples,), true values
    
    Returns:
        np.ndarray of shape (n_samples,), CQR scores
    """
    # Predict lower and upper quantiles for all points in X
    pred = qrf_model.predict(X, quantiles=[0.2, 0.8])  # Shape: (n_samples, 2)
    q_lower = pred[:, 0]  # Lower quantile predictions
    q_upper = pred[:, 1]  # Upper quantile predictions
    
    # Compute CQR score: max(q_lower - Y, Y - q_upper)
    scores = np.maximum(q_lower - Y, Y - q_upper)
    
    return scores