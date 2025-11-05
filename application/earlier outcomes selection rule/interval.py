import numpy as np
import math
from typing import Tuple, List, Callable
from config import method, q, reference_set_method
from selection import selection_rule_earlier_outcomes

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
    mu_on: np.ndarray,
    Y_off: np.ndarray,
    mu_off: np.ndarray,
    M: int,
    alpha: float,
    quantile_method: str = "upper",
    q: float = q,
    method: str = method,
    reference_set_method: str = reference_set_method
) -> Tuple[List[Tuple[float, float]], int]:
    """
    Build symmetric prediction interval at time t.
    This function works with selection_rule of form (X_j_val, cum_selected) -> bool
    or (mu_t, mu_history) -> bool, depending on the rule logic.
    """
    n_off = len(mu_off)
    #X_aug = np.concatenate((X_off,X_on))
    mu_aug = np.concatenate((mu_off,mu_on))
    Y_aug = np.concatenate((Y_off,Y_on))
    e=1e-10  # Approximate open interval: selection rule uses >, so bounds should be [lower, upper)

    if reference_set_method == "ours":
        if method == "quantile":
            if t+n_off == 0:
                return [(-math.inf, math.inf)], 0
            i = t
            indices_aug = list(range(i + n_off + 1))
            base_perm_aug = tuple(indices_aug)  # base_perm is the permutation of observed data

            cal_size = 0
            cal_size_lower = 0
            cal_size_upper = 0
            cal_size_middle = 0

            perms_list_aug = [base_perm_aug]
            max_perms_aug = min(M + 1, math.factorial(i + 1 + n_off))
            # perms_list is the list of initial random permutations

            while len(perms_list_aug) < max_perms_aug:
                pi = tuple(np.random.permutation(indices_aug))
                if pi not in perms_list_aug:
                    perms_list_aug.append(pi)

            # Get q-quantile of first i-1 points of y_on, plus values one above and one below
            y_hist = Y_aug[n_off:i+n_off]
            Q = int(np.ceil(i*q))  # Ensure Q is an integer
            y_hist_sorted = np.sort(y_hist)
            if Q <= 0 or Q >= i:
                return [(-math.inf, math.inf)], 0
            quantile = y_hist_sorted[Q-1]
            upper_quantile = y_hist_sorted[Q]
            lower_quantile = y_hist_sorted[Q-2]
            
            # First compute prediction set when y < lower quantile
            lower_quantile_prediction_set = []
            perms_selected_lower = []
            for pi in perms_list_aug:
                y_hist_temp = Y_aug[list(pi)][n_off:i+n_off]
                y_index_aug= pi.index(i+n_off)
                if y_index_aug>=n_off and y_index_aug<i+n_off:
                    y_hist_temp[y_index_aug] = lower_quantile-1
                #y_hist = y_hist_temp[n_off:i+n_off] 
                if selection_rule_earlier_outcomes(mu_aug[list(pi)][i+n_off], y_hist_temp,method=method,q=q):
                    perms_selected_lower.append(pi)
            cal_size_lower = len(perms_selected_lower)

            N1 = 0
            scores: List[float] = []
            for pi in perms_selected_lower:
                if pi[i+n_off] == i+n_off:
                    N1 += 1  # N1: we don't know exact score but know this permutation's score equals observed data's score
                else:
                    k = pi[i+n_off]
                    scores.append(abs(Y_aug[k] - mu_aug[k]))  # Use residual of the last point as score

            if not scores:
                lower_quantile_prediction_set.append((-math.inf, lower_quantile))
            else:
                scores.append(math.inf)  # Append inf to scores
                v_cut = get_smoothed_cutoff_with_N1(scores, alpha, N1, quantile_method=quantile_method)
                if mu_on[i]-v_cut >lower_quantile:
                    lower_quantile_prediction_set = []
                else:
                    lower_prediction_tuple=(mu_on[i]-v_cut, lower_quantile-e) if mu_on[i]+v_cut>lower_quantile else (mu_on[i]-v_cut, mu_on[i]+v_cut)
                    lower_quantile_prediction_set.append(lower_prediction_tuple)

            # Then compute prediction set when y > upper quantile
            upper_quantile_prediction_set = []
            perms_selected_upper = []
            for pi in perms_list_aug:
                y_hist_temp = Y_aug[list(pi)][n_off:i+n_off]
                y_index_aug= pi.index(i+n_off)
                if y_index_aug>=n_off and y_index_aug<i+n_off:
                    y_hist_temp[y_index_aug] = upper_quantile+1
                #y_hist = y_hist_temp[n_off:i+n_off] 
                if selection_rule_earlier_outcomes(mu_aug[list(pi)][i+n_off], y_hist_temp,method=method,q=q):
                    perms_selected_upper.append(pi)
            cal_size_upper = len(perms_selected_upper)  

            N1 = 0
            scores: List[float] = []
            for pi in perms_selected_upper:
                if pi[i+n_off] == i+n_off:
                    N1 += 1  # N1: we don't know exact score but know this permutation's score equals observed data's score
                else:
                    k = pi[i+n_off]
                    scores.append(abs(Y_aug[k] - mu_aug[k]))  # Use residual of the last point as score
            
            if not scores:
                upper_quantile_prediction_set.append((upper_quantile, math.inf))
            else:
                scores.append(math.inf)  # Append inf to scores
                v_cut = get_smoothed_cutoff_with_N1(scores, alpha, N1, quantile_method=quantile_method)
                if mu_on[i]+v_cut <upper_quantile:
                    upper_quantile_prediction_set = []  
                else:
                    upper_prediction_tuple=(mu_on[i]-v_cut, mu_on[i]+v_cut) if mu_on[i]-v_cut>upper_quantile else (upper_quantile, mu_on[i]+v_cut)
                    upper_quantile_prediction_set.append(upper_prediction_tuple)

            # Then compute prediction set when y is between lower and upper quantiles
            middle_quantile_prediction_set = []
            mu_middle=[]
            for mu in mu_on[:i]:
                if lower_quantile <= mu <= upper_quantile:
                    mu_middle.append(mu)
            middle_mu_intervals=[lower_quantile, upper_quantile]
            middle_mu_intervals_sorted = [lower_quantile, upper_quantile]
            if mu_middle:
                middle_mu_intervals.extend(mu_middle)
                middle_mu_intervals_sorted = np.sort(middle_mu_intervals)
            
            for mu_index in range(len(middle_mu_intervals_sorted)-1):
                lower_bound=middle_mu_intervals_sorted[mu_index]
                upper_bound=middle_mu_intervals_sorted[mu_index+1]
                perms_selected_middle = []
                for pi in perms_list_aug:
                    y_hist_temp = Y_aug[list(pi)][n_off:i+n_off]
                    y_index_aug= pi.index(i+n_off)
                    if y_index_aug>=n_off and y_index_aug<i+n_off:
                        y_hist_temp[y_index_aug] = (lower_bound+upper_bound)/2
                    #y_hist = y_hist_temp[n_off:i+n_off] 
                    if selection_rule_earlier_outcomes(mu_aug[list(pi)][i+n_off], y_hist_temp,method=method,q=q):
                        perms_selected_middle.append(pi)
                cal_size_middle = len(perms_selected_middle)
                N1 = 0
                scores: List[float] = []
                for pi in perms_selected_middle:
                    if pi[i+n_off] == i+n_off:
                        N1 += 1  # N1: we don't know exact score but know this permutation's score equals observed data's score
                    else:
                        k = pi[i+n_off]
                        scores.append(abs(Y_aug[k] - mu_aug[k]))  # Use residual of the last point as score   
                
                if not scores:
                    middle_quantile_prediction_set.append((lower_bound, upper_bound))
                else:
                    # e=1e-10 to approximate open interval: selection rule uses >, so bounds should be [lower, upper)
                    scores.append(math.inf)  # Append inf to scores
                    v_cut = get_smoothed_cutoff_with_N1(scores, alpha, N1, quantile_method=quantile_method)
                    if mu_on[i]-v_cut > upper_bound or mu_on[i]+v_cut < lower_bound:
                        continue
                    elif mu_on[i]-v_cut >lower_bound and mu_on[i]+v_cut<upper_bound:
                        middle_quantile_prediction_set.append((mu_on[i]-v_cut, mu_on[i]+v_cut))
                    elif mu_on[i]-v_cut<=lower_bound and mu_on[i]+v_cut>=upper_bound:
                        middle_quantile_prediction_set.append((lower_bound, upper_bound-e))
                    elif mu_on[i]-v_cut<=lower_bound and mu_on[i]+v_cut<upper_bound:
                        middle_quantile_prediction_set.append((lower_bound, mu_on[i]+v_cut))
                    elif mu_on[i]-v_cut>lower_bound and mu_on[i]+v_cut>=upper_bound:
                        middle_quantile_prediction_set.append((mu_on[i]-v_cut, upper_bound-e))
                    else:
                        middle_quantile_prediction_set.append((lower_bound, upper_bound-e))

            final_prediction_set = lower_quantile_prediction_set + upper_quantile_prediction_set + middle_quantile_prediction_set
            cal_size = cal_size_lower + cal_size_upper + cal_size_middle
            return final_prediction_set, cal_size
            
        

            
        elif method == "weighted_quantile":
            if t+n_off == 0:
                return [(-math.inf, math.inf)], 0
            i = t
            indices_aug = list(range(i + n_off + 1))
            base_perm_aug = tuple(indices_aug)  # base_perm is the permutation of observed data

            cal_size_weighted = 0

            perms_list_aug = [base_perm_aug]
            max_perms_aug = min(M + 1, math.factorial(i + 1 + n_off))
            # perms_list is the list of initial random permutations

            while len(perms_list_aug) < max_perms_aug:
                pi = tuple(np.random.permutation(indices_aug))
                if pi not in perms_list_aug:
                    perms_list_aug.append(pi)
            weighted_quantile_prediction_set = []
            mus_sorted=[-math.inf,math.inf]
            mus_sorted.extend(mu_aug[n_off:i+n_off])
            mus_sorted=np.sort(mus_sorted)
            for mu_index in range(len(mus_sorted)-1):
                lower_bound=mus_sorted[mu_index]
                upper_bound=mus_sorted[mu_index+1]
                perms_selected_weighted = []
                for pi in perms_list_aug:
                    y_hist_temp = Y_aug[list(pi)][n_off:i+n_off]
                    y_index_aug= pi.index(i+n_off)
                    #y_hist = y_hist_temp[n_off:i+n_off] 
                    if y_index_aug>=n_off and y_index_aug<i+n_off:
                        if lower_bound==-math.inf:
                            y_hist_temp[y_index_aug] = upper_bound-1  # Set y_t to upper_bound-1
                        elif upper_bound==math.inf:
                            y_hist_temp[y_index_aug] = lower_bound+1  # Set y_t to lower_bound+1
                        else:
                            y_hist_temp[y_index_aug] = (lower_bound+upper_bound)/2  # Set y_t to mean of lower_bound and upper_bound
                    if selection_rule_earlier_outcomes(mu_aug[list(pi)][i+n_off], y_hist_temp,method=method,q=q):
                        perms_selected_weighted.append(pi)
                cal_size_weighted = len(perms_selected_weighted)
                N1 = 0
                scores: List[float] = []
                for pi in perms_selected_weighted:
                    if pi[i+n_off] == i+n_off:
                        N1 += 1  # N1: we don't know exact score but know this permutation's score equals observed data's score
                    else:
                        k = pi[i+n_off]
                        scores.append(abs(Y_aug[k] - mu_aug[k]))  # Use residual of the last point as score   
                
                if not scores:
                    weighted_quantile_prediction_set.append((lower_bound, upper_bound))
                else:
                    e=1e-10  # Approximate open interval: selection rule uses >, so bounds should be [lower, upper)
                    scores.append(math.inf)  # Append inf to scores
                    v_cut = get_smoothed_cutoff_with_N1(scores, alpha, N1, quantile_method=quantile_method)
                    if mu_on[i]-v_cut > upper_bound or mu_on[i]+v_cut < lower_bound:
                        continue
                    elif mu_on[i]-v_cut >lower_bound and mu_on[i]+v_cut<upper_bound:
                        weighted_quantile_prediction_set.append((mu_on[i]-v_cut, mu_on[i]+v_cut))
                    elif mu_on[i]-v_cut<=lower_bound and mu_on[i]+v_cut>=upper_bound:
                        weighted_quantile_prediction_set.append((lower_bound, upper_bound-e))
                    elif mu_on[i]-v_cut<=lower_bound and mu_on[i]+v_cut<upper_bound:
                        weighted_quantile_prediction_set.append((lower_bound, mu_on[i]+v_cut))
                    elif mu_on[i]-v_cut>lower_bound and mu_on[i]+v_cut>=upper_bound:
                        weighted_quantile_prediction_set.append((mu_on[i]-v_cut, upper_bound-e))
                    else:
                        weighted_quantile_prediction_set.append((lower_bound, upper_bound-e))

            final_prediction_set = weighted_quantile_prediction_set
            cal_size = cal_size_weighted
            return final_prediction_set, cal_size

    elif reference_set_method == "vanilla":
        n_off = len(mu_off)
        mu_aug = np.concatenate((mu_off,mu_on))
        Y_aug = np.concatenate((Y_off,Y_on))
        if t+n_off == 0:
            return [(-math.inf, math.inf)], 0
        i = t
        N1 = 0
        cal_size = i+n_off

        scores: List[float] = []
        scores=list(np.abs(Y_aug[n_off:n_off+i]-mu_aug[n_off:n_off+i]))
        scores.append(math.inf)  # Append inf to scores

        v_cut = np.quantile(scores, 1-alpha, interpolation='higher')
        prediction_set = []
        prediction_set.append((mu_on[t] - v_cut, mu_on[t] + v_cut))

        return prediction_set, cal_size

    else:
        raise ValueError(f"Unknown method: {method!r}")
        
