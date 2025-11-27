# File: decision_permutation/interval.py
import numpy as np
import math
from typing import Tuple, List, Callable
from config import j_feature, method
from selection import selection_rule_optimization

def get_smoothed_cutoff_with_N1(scores: List[float], alpha: float, N1: int, quantile_method: str) -> float:
    """
    Compute p-value for a given v_test using randomized smoothing.
    p(y) = (#{scores > v_test} + U * (#{scores == v_test} + N1)) / (len(scores) + N1)
    """
    if quantile_method == "upper":
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
    mu_on_1: np.ndarray,
    mu_on_2: np.ndarray,
    mu_on_3: np.ndarray,
    Y_off: np.ndarray,
    mu_off_1: np.ndarray,
    mu_off_2: np.ndarray,
    mu_off_3: np.ndarray,
    M: int,
    alpha: float,
    quantile_method: str = "upper",
    reference_set_method: str = "ours"
) -> Tuple[float, float, int]:
    """
    Build symmetric prediction interval at time t.
    This function works with selection_rule of form (X_j_val, cum_selected) -> bool
    or (mu_t, mu_history) -> bool, depending on the rule logic.
    """
    if reference_set_method == "ours":
        n_off = len(mu_off_1)
        mu_aug_1 = np.concatenate((mu_off_1,mu_on_1))
        mu_aug_2 = np.concatenate((mu_off_2,mu_on_2))
        mu_aug_3 = np.concatenate((mu_off_3,mu_on_3))
        Y_aug = np.concatenate((Y_off,Y_on))
        if t+n_off == 0:
            return -math.inf, math.inf, 0
        i = t
        indices_aug = list(range(i + n_off + 1))
        base_perm_aug = tuple(indices_aug)  # base_perm is the permutation of observed data

        perms_list_aug = [base_perm_aug]
        max_perms_aug = min(M + 1, math.factorial(i + n_off + 1))
        # perms_list is the list of initial random permutations

        # Method 1: use simplest random permutation
        while len(perms_list_aug) < max_perms_aug:
            pi = tuple(np.random.permutation(indices_aug))
            if pi not in perms_list_aug:
                perms_list_aug.append(pi)

        perms_selected = []
        for pi in perms_list_aug:
            mu_pi_1 = mu_aug_1[list(pi)][n_off:i+n_off+1]  # From n_off to i, exclude offline data
            mu_pi_2 = mu_aug_2[list(pi)][n_off:i+n_off+1]
            mu_pi_3 = mu_aug_3[list(pi)][n_off:i+n_off+1]
            if selection_rule_optimization(mu_pi_1, mu_pi_2, mu_pi_3, method=method):
                perms_selected.append(pi)
        cal_size = len(perms_selected)

        N1 = 0
        scores: List[float] = []
        for pi in perms_selected:
            if pi[i+n_off] == i+n_off:
                N1 += 1  # N1: we don't know exact score but know this permutation's score equals observed data's score
            else:
                k = pi[i+n_off]
                scores.append(abs(Y_aug[k] - mu_aug_1[k]))  # Use residual of the last point as score

        if not scores:
            return -math.inf, math.inf, 0
        scores.append(math.inf)  # Append inf to scores

        v_cut = get_smoothed_cutoff_with_N1(scores, alpha, N1, quantile_method=quantile_method)
        return mu_on_1[t] - v_cut, mu_on_1[t] + v_cut, cal_size
    
    elif reference_set_method == "jomi":
        n_off = len(mu_off_1)
        mu_aug_1 = np.concatenate((mu_off_1,mu_on_1))
        mu_aug_2 = np.concatenate((mu_off_2,mu_on_2))
        mu_aug_3 = np.concatenate((mu_off_3,mu_on_3))
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
            mu_pi_1 = mu_aug_1[list(pi)][n_off:i+n_off+1]
            mu_pi_2 = mu_aug_2[list(pi)][n_off:i+n_off+1]
            mu_pi_3 = mu_aug_3[list(pi)][n_off:i+n_off+1]
            if selection_rule_optimization(mu_pi_1, mu_pi_2, mu_pi_3, method=method):
                perms_selected.append(pi)
        cal_size = len(perms_selected)

        N1 = 0
        scores: List[float] = []
        for pi in perms_selected:
            if pi[i+n_off] == i+n_off:
                N1 += 1  # N1: we don't know exact score but know this permutation's score equals observed data's score
            else:
                k = pi[i+n_off]
                scores.append(abs(Y_aug[k] - mu_aug_1[k]))  # Use residual of the last point as score

        if not scores:
            return -math.inf, math.inf, 0
        scores.append(math.inf)  # Append inf to scores

        v_cut = get_smoothed_cutoff_with_N1(scores, alpha, N1, quantile_method=quantile_method)
        return mu_on_1[t] - v_cut, mu_on_1[t] + v_cut, cal_size
    elif reference_set_method == "vanilla":
        n_off = len(mu_off_1)
        mu_aug_1 = np.concatenate((mu_off_1,mu_on_1))
        Y_aug = np.concatenate((Y_off,Y_on))
        if t+n_off == 0:
            return -math.inf, math.inf, 0
        i = t
        N1 = 0
        cal_size = i+n_off

        scores: List[float] = []
        scores=list(np.abs(Y_aug[n_off:n_off+i]-mu_aug_1[n_off:n_off+i]))
        scores.append(math.inf)  # Append inf to scores

        v_cut = np.quantile(scores, 1-alpha, interpolation='higher')

        return mu_on_1[t] - v_cut, mu_on_1[t] + v_cut, cal_size

