import numpy as np
import math
from typing import Tuple, List, Callable
from config import j_feature, method
from selection import selection_rule_conformal_p_value

def get_smoothed_cutoff_with_N1(scores: List[float], alpha: float, N1: int, quantile_method: str) -> float:
    """
    Compute p-value for a given v_test using randomized smoothing.
    p(y) = (#{scores > v_test} + U * (#{scores == v_test} + N1)) / (len(scores) + N1)
    """
    if quantile_method == "upper":
        scores_arr = np.array(scores) 
        total = len(scores_arr) + N1  # N1 includes scores from observed data permutations
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
    c_on: np.ndarray,
    w_i_on: np.ndarray,
    c_off: np.ndarray,
    w_i_off: np.ndarray,
    M: int,
    alpha: float,
    u: float,
    quantile_method: str = "upper",
    reference_set_method: str = "ours"
) -> Tuple[float, float, float, float, int]:
    """
    Build symmetric prediction interval at time t.
    This function works with selection_rule of form (X_j_val, cum_selected) -> bool
    or (mu_t, mu_history) -> bool, depending on the rule logic.
    """
    if reference_set_method == "ours":
        n_off = len(mu_off)
        mu_aug = np.concatenate((mu_off,mu_on))
        Y_aug = np.concatenate((Y_off,Y_on))
        c_aug = np.concatenate((c_off,c_on))
        w_i_aug = np.concatenate((w_i_off,w_i_on))
        if t+n_off == 0:
            return -math.inf, math.inf, -math.inf, math.inf, 0
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
        
        
        # Classify by y_t size: discuss y_t>c_t and y_t<=c_t cases separately
        perms_selected_k_0 = []
        perms_selected_k_1 = []
        for k in [0,1]:
            if k==0:
                for pi in perms_list_aug:
                    # Case y_t>c_t, k=0
                    y_index_aug=pi.index(t+n_off)
                    #if y_index_aug>=n_off:
                    y_index=y_index_aug-n_off
                    # else:
                    #     y_index=n_off-y_index_aug
                    mu_select = mu_aug[list(pi)][n_off:i+n_off+1]  # From n_off to i, exclude offline data
                    w_i_select = w_i_aug[list(pi)][n_off:i+n_off+1]
                    c_select = c_aug[list(pi)][n_off:i+n_off+1]
                    Y_select = Y_aug[list(pi)][n_off:i+n_off+1]
                    pi_select = tuple(list(range(i + 1)))

                    # Especially for e_lond
                    Y_cal=Y_aug[list(pi)][:n_off]
                    mu_cal=mu_aug[list(pi)][:n_off]
                    c_cal=c_aug[list(pi)][:n_off]
                    w_i_cal=w_i_aug[list(pi)][:n_off]

                    if selection_rule_conformal_p_value(w_i_select, c_select, Y_select, mu_select, Y_cal, mu_cal, c_cal, w_i_cal, pi_select, y_index, k, u, method=method):
                        perms_selected_k_0.append(pi)
            else:
                for pi in perms_list_aug:
                    # Case y_t<=c_t, k=1
                    y_index_aug=pi.index(t+n_off)
                    #if y_index_aug>=n_off:
                    y_index=y_index_aug-n_off
                    # else:
                    #     y_index=n_off-y_index_aug
                    mu_select = mu_aug[list(pi)][n_off:i+n_off+1]  # From n_off to i, exclude offline data
                    w_i_select = w_i_aug[list(pi)][n_off:i+n_off+1]
                    c_select = c_aug[list(pi)][n_off:i+n_off+1]
                    Y_select = Y_aug[list(pi)][n_off:i+n_off+1]
                    pi_select = tuple(list(range(i + 1)))

                    # Especially for e_lond
                    Y_cal=Y_aug[list(pi)][:n_off]
                    mu_cal=mu_aug[list(pi)][:n_off]
                    c_cal=c_aug[list(pi)][:n_off]
                    w_i_cal=w_i_aug[list(pi)][:n_off]

                    if selection_rule_conformal_p_value(w_i_select, c_select, Y_select, mu_select, Y_cal, mu_cal, c_cal, w_i_cal, pi_select, y_index, k, u, method=method):
                        perms_selected_k_1.append(pi)
        cal_size = (len(perms_selected_k_0) + len(perms_selected_k_1))/2



        N1 = 0
        scores_k_0 = []
        scores_k_1 = []
        for pi in perms_selected_k_0:
            if pi[i+n_off] == i+n_off:
                N1 += 1  # N1: we don't know exact score but know this permutation's score equals observed data's score
            else:
                k = pi[i+n_off]
                scores_k_0.append(abs(Y_aug[k] - mu_aug[k]))  # Use residual of the last point as score
        for pi in perms_selected_k_1:
            if pi[i+n_off] == i+n_off:
                N1 += 1  # N1: we don't know exact score but know this permutation's score equals observed data's score
            else:
                k = pi[i+n_off]
                scores_k_1.append(abs(Y_aug[k] - mu_aug[k]))  # Use residual of the last point as score

        if not scores_k_0:
            return -math.inf, math.inf, -math.inf, math.inf, 0
        scores_k_0.append(math.inf)  # Append inf to scores
        if not scores_k_1:
            return -math.inf, math.inf, -math.inf, math.inf, 0
        scores_k_1.append(math.inf)  # Append inf to scores

        v_cut_k_0 = get_smoothed_cutoff_with_N1(scores_k_0, alpha, N1, quantile_method=quantile_method)
        v_cut_k_1 = get_smoothed_cutoff_with_N1(scores_k_1, alpha, N1, quantile_method=quantile_method)

        if mu_on[t]-v_cut_k_1>c_on[t]:
            bound_1=0
            bound_2=0
        else:
            bound_1=mu_on[t]-v_cut_k_1
            if mu_on[t]+v_cut_k_1<c_on[t]:
                bound_2=mu_on[t]+v_cut_k_1
            else:
                bound_2=c_on[t]
        if mu_on[t]+v_cut_k_0<c_on[t]:
            bound_3=0
            bound_4=0
        else:
            bound_4=mu_on[t]+v_cut_k_0
            if mu_on[t]-v_cut_k_0>c_on[t]:
                bound_3=mu_on[t]-v_cut_k_0
            else:
                bound_3=c_on[t]

        return bound_1, bound_2, bound_3, bound_4, cal_size
    elif reference_set_method == "vanilla":
        n_off = len(mu_off)
        mu_aug = np.concatenate((mu_off,mu_on))
        Y_aug = np.concatenate((Y_off,Y_on))
        if t+n_off == 0:
            return -math.inf, math.inf, -math.inf, math.inf, 0
        i = t
        N1 = 0
        cal_size = i+n_off

        scores: List[float] = []
        scores=list(np.abs(Y_aug[n_off:n_off+i]-mu_aug[n_off:n_off+i]))
        scores.append(math.inf)  # Append inf to scores 

        v_cut = np.quantile(scores, 1-alpha, interpolation='higher')

        return mu_on[t] - v_cut, mu_on[t] + v_cut, mu_on[t], mu_on[t], cal_size
