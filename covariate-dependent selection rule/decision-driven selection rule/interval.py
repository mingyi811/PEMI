import numpy as np
import math
from typing import Tuple, List, Callable
from config import j_feature, reference_set_method
from selection import selection_rule, selection_rule_2, selection_rule_final


def get_smoothed_cutoff_with_N1(scores: List[float], alpha: float, N1: int, method: str) -> float:
    """
    Compute p-value for a given v_test using randomized smoothing.
    p(y) = (#{scores > v_test} + U * (#{scores == v_test} + N1)) / (len(scores) + N1)
    """
    # Method 1: upper quantile cutoff (valid but larger)
    if method == "upper":
        scores_arr = np.array(scores)  
        total = len(scores_arr) + N1  # N1 includes scores same as observed data
        v_upper = math.inf
        for v in np.sort(np.unique(scores_arr))[::-1]:
            num_gt = np.sum(scores_arr > v)
            num_eq = np.sum(scores_arr == v)
            p_val = (num_gt + (num_eq + N1)) / (total)
            if p_val > alpha:  
                return v_upper  # Return the last v where pvalue <= alpha
            v_upper = v
        return v_upper

    # Method 2: randomized cutoff 
    elif method == "randomize":
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
        raise ValueError(f"Unknown method: {method!r}")



def construct_prediction_interval(
    t: int,
    Y_on: np.ndarray,
    mu_on: np.ndarray,
    Y_off: np.ndarray,
    mu_off: np.ndarray,
    history: List[bool],
    M: int,
    alpha: float,
    quantile_method: str = "upper",
    reference_set_method: str = reference_set_method
) -> Tuple[float, float, int]:
    """
    Build prediction interval at time t via randomized permutations.
    """
    if reference_set_method == "ours":
        n_off = len(mu_off)
        #X_aug = np.concatenate((X_off,X_on))
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
        # X_slice = X_aug[: i + n_off + 1]  # X_slice is observed data's X
        mu_slice = mu_aug[:i+n_off+1]
        for pi in perms_list_aug:
            cum=0
            for k in range(i):
                if selection_rule_final(mu_slice[pi[k+n_off]], cum):
                    cum += 1
            if selection_rule_final(mu_slice[pi[i+n_off]], cum):
                perms_selected.append(pi)
            # cum = 0
            # for k in range(i):
            #     if selection_rule(X_slice[pi[k+n_off], j_feature], cum):
            #         cum += 1
            # if selection_rule(X_slice[pi[i+n_off], j_feature], cum):
            #     perms_selected.append(pi)
            # cum=0
            # for k in range(i):
            #     if selection_rule_2(X_slice[pi[k+n_off], j_feature], cum, k, i):
            #         cum += 1
            # if selection_rule_2(X_slice[pi[i+n_off], j_feature], cum, i, i):
            #     perms_selected.append(pi)
        cal_size = len(perms_selected)

        # Separate observed vs. visible residuals
        N1 = 0
        scores: List[float] = []
        for pi in perms_selected:
            if pi[i+n_off] == i+n_off:
                N1 += 1  # N1: we don't know exact score but know this permutation's score equals observed data's score
            else:
                k = pi[i+n_off]
                scores.append(float(np.abs(Y_aug[k] - mu_aug[k])))  # Use residual of the last point as score

        if not scores:
            return -math.inf, math.inf, 0

        scores.append(float(np.inf))  # Append inf to scores

        v_cut = get_smoothed_cutoff_with_N1(scores, alpha, N1, quantile_method)
        return mu_on[t] - v_cut, mu_on[t] + v_cut, cal_size

    elif reference_set_method == "jomi":
        n_off = len(mu_off)
        #X_aug = np.concatenate((X_off,X_on))
        mu_aug = np.concatenate((mu_off,mu_on))
        Y_aug = np.concatenate((Y_off,Y_on))
        if t+n_off == 0:
            return -math.inf, math.inf, 0
        i = t
        indices_aug = list(range(i + n_off + 1))
        base_perm_aug = tuple(indices_aug)  # base_perm is the permutation of observed data

        perms_list_aug = [base_perm_aug]

        for j in range(i+n_off):
            pi = list(base_perm_aug)
            pi[j], pi[i+n_off] = pi[i+n_off], pi[j]
            perms_list_aug.append(tuple(pi))

        perms_selected = []
        mu_slice = mu_aug[: i + n_off + 1]
        for pi in perms_list_aug:
            # cum = 0
            # for k in range(i):
            #     if selection_rule(X_slice[pi[k+n_off], j_feature], cum):
            #         cum += 1
            # if selection_rule(X_slice[pi[i+n_off], j_feature], cum):
            #     perms_selected.append(pi)
            cum=0
            for k in range(i):
                if selection_rule_final(mu_slice[pi[k+n_off]], cum):
                    cum += 1
            if selection_rule_final(mu_slice[pi[i+n_off]], cum):
                perms_selected.append(pi)
        cal_size = len(perms_selected)

        N1 = 0
        scores: List[float] = []
        for pi in perms_selected:
            if pi[i+n_off] == i+n_off:
                N1 += 1  # N1: we don't know exact score but know this permutation's score equals observed data's score
            else:
                k = pi[i+n_off]
                scores.append(float(np.abs(Y_aug[k] - mu_aug[k])))  # Use residual of the last point as score

        if not scores:
            return -math.inf, math.inf, 0

        scores.append(float(np.inf))  # Append inf to scores
        

        v_cut = get_smoothed_cutoff_with_N1(scores, alpha, N1, quantile_method)
        return mu_on[t] - v_cut, mu_on[t] + v_cut, cal_size

    elif reference_set_method == "vanilla":
        n_off = len(mu_off)
        mu_aug = np.concatenate((mu_off,mu_on))
        Y_aug = np.concatenate((Y_off,Y_on))
        if t+n_off == 0:
            return -math.inf, math.inf, 0
        i = t
        N1 = 0
        cal_size = i+n_off

        scores: List[float] = []
        scores=list(np.abs(Y_aug[n_off:n_off+i]-mu_aug[n_off:n_off+i]))
        scores.append(math.inf)  # Append inf to scores

        v_cut = np.quantile(scores, 1-alpha, interpolation='higher')

        return mu_on[t] - v_cut, mu_on[t] + v_cut, cal_size
    elif reference_set_method == "express":
        n_off = len(mu_off)
        mu_aug = np.concatenate((mu_off,mu_on))
        Y_aug = np.concatenate((Y_off,Y_on))
        if t+n_off == 0:
            return -math.inf, math.inf, 0
        i = t
        N1 = 0
        scores: List[float] = []

        for j in range(t):
            # Agree on all past selection rules
            agree = True
            for i in range(t):
                Si_on_Xj = selection_rule_final(mu_aug[j], np.sum(history[:i]))
                Si_on_Xt = selection_rule_final(mu_aug[t], np.sum(history[:i]))
                if Si_on_Xj != Si_on_Xt:
                    agree = False
                    break
            # St(X_j) = 1
            if agree and selection_rule_final(mu_aug[j], np.sum(history)):
                scores.append(float(np.abs(Y_aug[j] - mu_aug[j])))

        if not scores:
            return -math.inf, math.inf, 0
        
        cal_size = len(scores)

        scores.append(float(np.inf))  # Append inf to scores

        v_cut = np.quantile(scores, 1-alpha, interpolation='higher')
        return mu_on[t] - v_cut, mu_on[t] + v_cut, cal_size


