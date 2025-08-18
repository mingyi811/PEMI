# File: decision_permutation/interval.py
import numpy as np
import math
from typing import Tuple, List, Callable
from config import method, q
from selection import selection_rule_earlier_outcomes

def get_smoothed_cutoff_with_N1(scores: List[float], alpha: float, N1: int, quantile_method: str) -> float:
    """
    Compute p-value for a given v_test using randomized smoothing.
    p(y) = (#{scores > v_test} + U * (#{scores == v_test} + N1)) / (len(scores) + N1)
    """
    if quantile_method == "upper":
        scores_arr = np.array(scores) #scores中有append inf
        total = len(scores_arr) + N1 #N1中包含observed data的permutation的score
        v_upper = math.inf
        for v in np.sort(np.unique(scores_arr))[::-1]:
            num_gt = np.sum(scores_arr > v)
            num_eq = np.sum(scores_arr == v)
            p_val = (num_gt + (num_eq + N1)) / (total)
            if p_val >= alpha*(1+1/total): #这里加1/total是因为我们append了一个inf  
                return v_upper #返回的是最后一个让pvalue<alpha的v
            v_upper = v
        return v_upper

    #方案2: randomize的cutoff(valid但比方案一好一点)
    elif quantile_method == "randomize":
        scores_arr = np.array(scores)
        total = len(scores_arr) + N1
        U=np.random.uniform(0,1)
        v_upper = math.inf
        for v in np.sort(np.unique(scores_arr))[::-1]:
            num_gt = np.sum(scores_arr > v)
            num_eq = np.sum(scores_arr == v)
            p_val = (num_gt + U * (num_eq + N1)) / (total)
            if p_val >= alpha:
                return v_upper
            v_upper = v
        return v_upper
    else:
        raise ValueError(f"Unknown method: {quantile_method!r}")

    #方案2: 我自己算的那个cutoff
    # scores_arr = np.array(scores)
    # total = len(scores_arr) + N1
    # U=np.random.uniform(0,1) 
    # v_upper = math.inf
    # num_gt_upper = 0
    # num_eq_upper = 0    
    # #index_upper = total
    # v_list = np.sort(np.unique(scores_arr))[::-1]
    # for idx, v in enumerate(v_list):
    #     num_gt = np.sum(scores_arr > v)
    #     num_eq = np.sum(scores_arr == v)
    #     # k: rank of v in total (from 0 for largest)
    #     #index_lower = idx
    #     #f = (total*alpha+index_upper-total-0.5*(N1+1)-N1)/(index_upper-index_lower)
    #     prob = (alpha*total-0.5*N1-num_gt_upper-0.5*num_eq_upper)/(num_gt+0.5*num_eq-num_gt_upper-0.5*num_eq_upper)
    #     p_val = (num_gt + U * (num_eq + N1)) / (total)
    #     if p_val >= alpha:
    #         if U < prob:
    #             return v
    #         else:
    #             return v_upper
    #     v_upper = v
    #     #index_upper = idx
    #     num_gt_upper = num_gt
    #     num_eq_upper = num_eq
    # return math.inf

def construct_prediction_interval(
    t: int,
    X_on: np.ndarray,
    Y_on: np.ndarray,
    mu_on: np.ndarray,
    X_off: np.ndarray,
    Y_off: np.ndarray,
    mu_off: np.ndarray,
    M: int,
    alpha: float,
    quantile_method: str = "upper",
    q: float = q,
    method: str = method
) -> Tuple[List[Tuple[float, float]], int]:
    """
    Build symmetric prediction interval at time t.
    This function works with selection_rule of form (X_j_val, cum_selected) -> bool
    or (mu_t, mu_history) -> bool, depending on the rule logic.
    """
    n_off = len(X_off)
    X_aug = np.concatenate((X_off,X_on))
    mu_aug = np.concatenate((mu_off,mu_on))
    Y_aug = np.concatenate((Y_off,Y_on))
    if method == "quantile":
        if t+n_off == 0:
            return [(-math.inf, math.inf)], 0
        i = t
        indices_aug = list(range(i + n_off + 1))
        base_perm_aug = tuple(indices_aug) #base_perm是observed data的permutation

        cal_size = 0
        cal_size_lower = 0
        cal_size_upper = 0
        cal_size_middle = 0

        perms_list_aug = [base_perm_aug]
        max_perms_aug = min(M + 1, math.factorial(i + 1 + n_off))
        #perms_list是所有最开始random permutation的list

        while len(perms_list_aug) < max_perms_aug:
            pi = tuple(np.random.permutation(indices_aug))
            if pi not in perms_list_aug:
                perms_list_aug.append(pi)

        #获取y_on的前i-1个点的q-quantile以及比它大一位和比它小一位的数值
        y_hist = Y_aug[n_off:i+n_off]
        Q = int(np.ceil(i*q))  # 确保 Q 是整数
        y_hist_sorted = np.sort(y_hist)
        if Q <= 0 or Q >= i:
            return [(-math.inf, math.inf)], 0
        quantile = y_hist_sorted[Q-1]
        upper_quantile = y_hist_sorted[Q]
        lower_quantile = y_hist_sorted[Q-2]
        
        #先计算当y小于lower quantile的时候的prediction set
        lower_quantile_prediction_set = []
        perms_selected_lower = []
        for pi in perms_list_aug:
            y_index_aug= pi.index(i+n_off)
            if y_index_aug>=n_off and y_index_aug<i+n_off:
                Y_aug[y_index_aug] = lower_quantile-1
            y_hist = Y_aug[list(pi)][n_off:i+n_off] 
            if selection_rule_earlier_outcomes(mu_aug[list(pi)][i+n_off], y_hist,method=method,q=q):
                perms_selected_lower.append(pi)
        cal_size_lower = len(perms_selected_lower)

        N1 = 0
        scores: List[float] = []
        for pi in perms_selected_lower:
            if pi[i+n_off] == i+n_off:
                N1 += 1 #N1是我们不知道具体score但知道这个permutation的score是等于observed data的score的
            else:
                k = pi[i+n_off]
                scores.append(abs(Y_aug[k] - mu_aug[k])) #最后一个点的residual作为score

        if not scores:
            lower_quantile_prediction_set.append((-math.inf, lower_quantile))
        else:
            scores.append(math.inf) #scores中append inf
            v_cut = get_smoothed_cutoff_with_N1(scores, alpha, N1, quantile_method=quantile_method)
            if mu_on[i]-v_cut >lower_quantile:
                lower_quantile_prediction_set = []
            else:
                lower_prediction_tuple=(mu_on[i]-v_cut, lower_quantile) if mu_on[i]+v_cut>lower_quantile else (mu_on[i]-v_cut, mu_on[i]+v_cut)
                lower_quantile_prediction_set.append(lower_prediction_tuple)

        #再计算当y大于upper quantile的时候的prediction set
        upper_quantile_prediction_set = []
        perms_selected_upper = []
        for pi in perms_list_aug:
            y_index_aug= pi.index(i+n_off)
            if y_index_aug>=n_off and y_index_aug<i+n_off:
                Y_aug[y_index_aug] = upper_quantile+1
            y_hist = Y_aug[list(pi)][n_off:i+n_off] 
            if selection_rule_earlier_outcomes(mu_aug[list(pi)][i+n_off], y_hist,method=method,q=q):
                perms_selected_upper.append(pi)
        cal_size_upper = len(perms_selected_upper)  

        N1 = 0
        scores: List[float] = []
        for pi in perms_selected_upper:
            if pi[i+n_off] == i+n_off:
                N1 += 1 #N1是我们不知道具体score但知道这个permutation的score是等于observed data的score的
            else:
                k = pi[i+n_off]
                scores.append(abs(Y_aug[k] - mu_aug[k])) #最后一个点的residual作为score
        
        if not scores:
            upper_quantile_prediction_set.append((upper_quantile, math.inf))
        else:
            scores.append(math.inf) #scores中append inf
            v_cut = get_smoothed_cutoff_with_N1(scores, alpha, N1, quantile_method=quantile_method)
            if mu_on[i]+v_cut <upper_quantile:
                upper_quantile_prediction_set = []  
            else:
                upper_prediction_tuple=(mu_on[i]-v_cut, mu_on[i]+v_cut) if mu_on[i]-v_cut>upper_quantile else (upper_quantile, mu_on[i]+v_cut)
                upper_quantile_prediction_set.append(upper_prediction_tuple)

        #再计算当y在lower quantile和upper quantile之间的时候的prediction set
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
                y_index_aug= pi.index(i+n_off)
                if y_index_aug>=n_off and y_index_aug<i+n_off:
                    Y_aug[y_index_aug] = (lower_bound+upper_bound)/2
                y_hist = Y_aug[list(pi)][n_off:i+n_off] 
                if selection_rule_earlier_outcomes(mu_aug[list(pi)][i+n_off], y_hist,method=method,q=q):
                    perms_selected_middle.append(pi)
            cal_size_middle = len(perms_selected_middle)
            N1 = 0
            scores: List[float] = []
            for pi in perms_selected_middle:
                if pi[i+n_off] == i+n_off:
                    N1 += 1 #N1是我们不知道具体score但知道这个permutation的score是等于observed data的score的
                else:
                    k = pi[i+n_off]
                    scores.append(abs(Y_aug[k] - mu_aug[k])) #最后一个点的residual作为score   
            
            if not scores:
                middle_quantile_prediction_set.append((lower_bound, upper_bound))
            else:
                scores.append(math.inf) #scores中append inf
                v_cut = get_smoothed_cutoff_with_N1(scores, alpha, N1, quantile_method=quantile_method)
                if mu_on[i]-v_cut > upper_bound or mu_on[i]+v_cut < lower_bound:
                    continue
                elif mu_on[i]-v_cut >lower_bound and mu_on[i]+v_cut<upper_bound:
                    middle_quantile_prediction_set.append((mu_on[i]-v_cut, mu_on[i]+v_cut))
                elif mu_on[i]-v_cut<=lower_bound and mu_on[i]+v_cut>=upper_bound:
                    middle_quantile_prediction_set.append((lower_bound, upper_bound))
                elif mu_on[i]-v_cut<=lower_bound and mu_on[i]+v_cut<upper_bound:
                    middle_quantile_prediction_set.append((lower_bound, mu_on[i]+v_cut))
                elif mu_on[i]-v_cut>lower_bound and mu_on[i]+v_cut>=upper_bound:
                    middle_quantile_prediction_set.append((mu_on[i]-v_cut, upper_bound))
                else:
                    middle_quantile_prediction_set.append((lower_bound, upper_bound))
        

        final_prediction_set = lower_quantile_prediction_set + upper_quantile_prediction_set + middle_quantile_prediction_set
        cal_size = cal_size_lower + cal_size_upper + cal_size_middle
        return final_prediction_set, cal_size
        
     

        
    elif method == "weighted_quantile":
        if t+n_off == 0:
            return [(-math.inf, math.inf)], 0
        i = t
        indices_aug = list(range(i + n_off + 1))
        base_perm_aug = tuple(indices_aug) #base_perm是observed data的permutation

        cal_size_weighted = 0

        perms_list_aug = [base_perm_aug]
        max_perms_aug = min(M + 1, math.factorial(i + 1 + n_off))
        #perms_list是所有最开始random permutation的list

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
                y_index_aug= pi.index(i+n_off)
                y_hist = Y_aug[list(pi)][n_off:i+n_off] 
                if y_index_aug>=n_off and y_index_aug<i+n_off:
                    if lower_bound==-math.inf:
                        Y_aug[y_index_aug] = upper_bound-1 #把y_t设置为upper_bound-1
                    elif upper_bound==math.inf:
                        Y_aug[y_index_aug] = lower_bound+1 #把y_t设置为lower_bound+1
                    else:
                        Y_aug[y_index_aug] = (lower_bound+upper_bound)/2 #把y_t设置为lower_bound和upper_bound的均值
                if selection_rule_earlier_outcomes(mu_aug[list(pi)][i+n_off], y_hist,method=method,q=q):
                    perms_selected_weighted.append(pi)
            cal_size_weighted = len(perms_selected_weighted)
            N1 = 0
            scores: List[float] = []
            for pi in perms_selected_weighted:
                if pi[i+n_off] == i+n_off:
                    N1 += 1 #N1是我们不知道具体score但知道这个permutation的score是等于observed data的score的
                else:
                    k = pi[i+n_off]
                    scores.append(abs(Y_aug[k] - mu_aug[k])) #最后一个点的residual作为score   
            
            if not scores:
                weighted_quantile_prediction_set.append((lower_bound, upper_bound))
            else:
                scores.append(math.inf) #scores中append inf
                v_cut = get_smoothed_cutoff_with_N1(scores, alpha, N1, quantile_method=quantile_method)
                if mu_on[i]-v_cut > upper_bound or mu_on[i]+v_cut < lower_bound:
                    continue
                elif mu_on[i]-v_cut >lower_bound and mu_on[i]+v_cut<upper_bound:
                    weighted_quantile_prediction_set.append((mu_on[i]-v_cut, mu_on[i]+v_cut))
                elif mu_on[i]-v_cut<=lower_bound and mu_on[i]+v_cut>=upper_bound:
                    weighted_quantile_prediction_set.append((lower_bound, upper_bound))
                elif mu_on[i]-v_cut<=lower_bound and mu_on[i]+v_cut<upper_bound:
                    weighted_quantile_prediction_set.append((lower_bound, mu_on[i]+v_cut))
                elif mu_on[i]-v_cut>lower_bound and mu_on[i]+v_cut>=upper_bound:
                    weighted_quantile_prediction_set.append((mu_on[i]-v_cut, upper_bound))
                else:
                    weighted_quantile_prediction_set.append((lower_bound, upper_bound))

        return weighted_quantile_prediction_set, cal_size_weighted

    else:
        raise ValueError(f"Unknown method: {method!r}")
        




# def construct_prediction_interval(
#     t: int,
#     X_on: np.ndarray,
#     Y_on: np.ndarray,
#     mu_on: np.ndarray,
#     M: int,
#     alpha: float,
#     quantile_method: str = "upper",
#     q: float = q,
#     method: str = method
# ) -> Tuple[List[Tuple[float, float]], int]:
#     """
#     Build symmetric prediction interval at time t.
#     This function works with selection_rule of form (X_j_val, cum_selected) -> bool
#     or (mu_t, mu_history) -> bool, depending on the rule logic.
#     """
#     if method == "quantile":
#         if t == 0:
#             return [(-math.inf, math.inf)], 0
#         i = t
#         indices = list(range(i + 1))
#         base_perm = tuple(indices) #base_perm是observed data的permutation

#         cal_size = 0
#         cal_size_lower = 0
#         cal_size_upper = 0
#         cal_size_middle = 0

#         perms_list = [base_perm]
#         max_perms = min(M + 1, math.factorial(i + 1))
#         #perms_list是所有最开始random permutation的list

#         while len(perms_list) < max_perms:
#             pi = tuple(np.random.permutation(indices))
#             if pi not in perms_list:
#                 perms_list.append(pi)

#         #获取y_on的前i-1个点的q-quantile以及比它大一位和比它小一位的数值
#         y_hist = Y_on[:i]
#         Q = int(np.ceil(i*q))  # 确保 Q 是整数
#         y_hist_sorted = np.sort(y_hist)
#         if Q <= 0 or Q >= i:
#             return [(-math.inf, math.inf)], 0
#         quantile = y_hist_sorted[Q-1]
#         upper_quantile = y_hist_sorted[Q]
#         lower_quantile = y_hist_sorted[Q-2]
        
#         #先计算当y小于lower quantile的时候的prediction set
#         lower_quantile_prediction_set = []
#         perms_selected_lower = []
#         for pi in perms_list_aug:
#             y_index= pi.index(i)
#             y_hist = Y_on[list(pi)][:i] 
#             if y_index < i:
#                 y_hist[y_index] = lower_quantile-1 #把y_t设置为lower quantile-1
#             if selection_rule_earlier_outcomes(mu_on[list(pi)][i], y_hist,method=method,q=q):
#                 perms_selected_lower.append(pi)
#         cal_size_lower = len(perms_selected_lower)

#         N1 = 0
#         scores: List[float] = []
#         for pi in perms_selected_lower:
#             if pi[i] == i:
#                 N1 += 1 #N1是我们不知道具体score但知道这个permutation的score是等于observed data的score的
#             else:
#                 k = pi[i]
#                 scores.append(abs(Y_on[k] - mu_on[k])) #最后一个点的residual作为score

#         if not scores:
#             lower_quantile_prediction_set.append((-math.inf, lower_quantile))
#         else:
#             scores.append(math.inf) #scores中append inf
#             v_cut = get_smoothed_cutoff_with_N1(scores, alpha, N1, quantile_method=quantile_method)
#             if mu_on[i]-v_cut >lower_quantile:
#                 lower_quantile_prediction_set = []
#             else:
#                 lower_prediction_tuple=(mu_on[i]-v_cut, lower_quantile) if mu_on[i]+v_cut>lower_quantile else (mu_on[i]-v_cut, mu_on[i]+v_cut)
#                 lower_quantile_prediction_set.append(lower_prediction_tuple)

#         #再计算当y大于upper quantile的时候的prediction set
#         upper_quantile_prediction_set = []
#         perms_selected_upper = []
#         for pi in perms_list:
#             y_index= pi.index(i)
#             y_hist = Y_on[list(pi)][:i] 
#             if y_index < i:
#                 y_hist[y_index] = upper_quantile+1 #把y_t设置为upper quantile+1
#             if selection_rule_earlier_outcomes(mu_on[list(pi)][i], y_hist,method=method,q=q):
#                 perms_selected_upper.append(pi)
#         cal_size_upper = len(perms_selected_upper)  

#         N1 = 0
#         scores: List[float] = []
#         for pi in perms_selected_upper:
#             if pi[i] == i:
#                 N1 += 1 #N1是我们不知道具体score但知道这个permutation的score是等于observed data的score的
#             else:
#                 k = pi[i]
#                 scores.append(abs(Y_on[k] - mu_on[k])) #最后一个点的residual作为score
        
#         if not scores:
#             upper_quantile_prediction_set.append((upper_quantile, math.inf))
#         else:
#             scores.append(math.inf) #scores中append inf
#             v_cut = get_smoothed_cutoff_with_N1(scores, alpha, N1, quantile_method=quantile_method)
#             if mu_on[i]+v_cut <upper_quantile:
#                 upper_quantile_prediction_set = []  
#             else:
#                 upper_prediction_tuple=(mu_on[i]-v_cut, mu_on[i]+v_cut) if mu_on[i]-v_cut>upper_quantile else (upper_quantile, mu_on[i]+v_cut)
#                 upper_quantile_prediction_set.append(upper_prediction_tuple)

#         #再计算当y在lower quantile和upper quantile之间的时候的prediction set
#         middle_quantile_prediction_set = []
#         mu_middle=[]
#         for mu in mu_on[:i]:
#             if lower_quantile <= mu <= upper_quantile:
#                 mu_middle.append(mu)
#         middle_mu_intervals=[lower_quantile, upper_quantile]
#         middle_mu_intervals_sorted = [lower_quantile, upper_quantile]
#         if mu_middle:
#             middle_mu_intervals.extend(mu_middle)
#             middle_mu_intervals_sorted = np.sort(middle_mu_intervals)
        
#         for mu_index in range(len(middle_mu_intervals_sorted)-1):
#             lower_bound=middle_mu_intervals_sorted[mu_index]
#             upper_bound=middle_mu_intervals_sorted[mu_index+1]
#             perms_selected_middle = []
#             for pi in perms_list:
#                 y_index= pi.index(i)
#                 y_hist = Y_on[list(pi)][:i] 
#                 if y_index < i:
#                     y_hist[y_index] = (lower_bound+upper_bound)/2 #把y_t设置为lower_bound和upper_bound的均值
#                 if selection_rule_earlier_outcomes(mu_on[list(pi)][i], y_hist,method=method,q=q):
#                     perms_selected_middle.append(pi)
#             cal_size_middle = len(perms_selected_middle)
#             N1 = 0
#             scores: List[float] = []
#             for pi in perms_selected_middle:
#                 if pi[i] == i:
#                     N1 += 1 #N1是我们不知道具体score但知道这个permutation的score是等于observed data的score的
#                 else:
#                     k = pi[i]
#                     scores.append(abs(Y_on[k] - mu_on[k])) #最后一个点的residual作为score   
            
#             if not scores:
#                 middle_quantile_prediction_set.append((lower_bound, upper_bound))
#             else:
#                 scores.append(math.inf) #scores中append inf
#                 v_cut = get_smoothed_cutoff_with_N1(scores, alpha, N1, quantile_method=quantile_method)
#                 if mu_on[i]-v_cut > upper_bound or mu_on[i]+v_cut < lower_bound:
#                     continue
#                 elif mu_on[i]-v_cut >lower_bound and mu_on[i]+v_cut<upper_bound:
#                     middle_quantile_prediction_set.append((mu_on[i]-v_cut, mu_on[i]+v_cut))
#                 elif mu_on[i]-v_cut<=lower_bound and mu_on[i]+v_cut>=upper_bound:
#                     middle_quantile_prediction_set.append((lower_bound, upper_bound))
#                 elif mu_on[i]-v_cut<=lower_bound and mu_on[i]+v_cut<upper_bound:
#                     middle_quantile_prediction_set.append((lower_bound, mu_on[i]+v_cut))
#                 elif mu_on[i]-v_cut>lower_bound and mu_on[i]+v_cut>=upper_bound:
#                     middle_quantile_prediction_set.append((mu_on[i]-v_cut, upper_bound))
#                 else:
#                     middle_quantile_prediction_set.append((lower_bound, upper_bound))
        

#         final_prediction_set = lower_quantile_prediction_set + upper_quantile_prediction_set + middle_quantile_prediction_set
#         cal_size = cal_size_lower + cal_size_upper + cal_size_middle
#         return final_prediction_set, cal_size
        
     

        
#     elif method == "weighted_quantile":
#         if t == 0:
#             return [(-math.inf, math.inf)], 0
#         i = t
#         indices = list(range(i + 1))
#         base_perm = tuple(indices) #base_perm是observed data的permutation

#         cal_size_weighted = 0

#         perms_list = [base_perm]
#         max_perms = min(M + 1, math.factorial(i + 1))
#         #perms_list是所有最开始random permutation的list

#         while len(perms_list) < max_perms:
#             pi = tuple(np.random.permutation(indices))
#             if pi not in perms_list:
#                 perms_list.append(pi)
#         weighted_quantile_prediction_set = []
#         mus_sorted=[-math.inf,math.inf]
#         mus_sorted.extend(mu_on[:t])
#         mus_sorted=np.sort(mus_sorted)
#         for mu_index in range(len(mus_sorted)-1):
#             lower_bound=mus_sorted[mu_index]
#             upper_bound=mus_sorted[mu_index+1]
#             perms_selected_weighted = []
#             for pi in perms_list:
#                 y_index= pi.index(i)
#                 y_hist = Y_on[list(pi)][:i] 
#                 if y_index < i:
#                     if lower_bound==-math.inf:
#                         y_hist[y_index] = upper_bound-1 #把y_t设置为upper_bound-1
#                     elif upper_bound==math.inf:
#                         y_hist[y_index] = lower_bound+1 #把y_t设置为lower_bound+1
#                     else:
#                         y_hist[y_index] = (lower_bound+upper_bound)/2 #把y_t设置为lower_bound和upper_bound的均值
#                 if selection_rule_earlier_outcomes(mu_on[list(pi)][i], y_hist,method=method,q=q):
#                     perms_selected_weighted.append(pi)
#             cal_size_weighted = len(perms_selected_weighted)
#             N1 = 0
#             scores: List[float] = []
#             for pi in perms_selected_weighted:
#                 if pi[i] == i:
#                     N1 += 1 #N1是我们不知道具体score但知道这个permutation的score是等于observed data的score的
#                 else:
#                     k = pi[i]
#                     scores.append(abs(Y_on[k] - mu_on[k])) #最后一个点的residual作为score   
            
#             if not scores:
#                 weighted_quantile_prediction_set.append((lower_bound, upper_bound))
#             else:
#                 scores.append(math.inf) #scores中append inf
#                 v_cut = get_smoothed_cutoff_with_N1(scores, alpha, N1, quantile_method=quantile_method)
#                 if mu_on[i]-v_cut > upper_bound or mu_on[i]+v_cut < lower_bound:
#                     continue
#                 elif mu_on[i]-v_cut >lower_bound and mu_on[i]+v_cut<upper_bound:
#                     weighted_quantile_prediction_set.append((mu_on[i]-v_cut, mu_on[i]+v_cut))
#                 elif mu_on[i]-v_cut<=lower_bound and mu_on[i]+v_cut>=upper_bound:
#                     weighted_quantile_prediction_set.append((lower_bound, upper_bound))
#                 elif mu_on[i]-v_cut<=lower_bound and mu_on[i]+v_cut<upper_bound:
#                     weighted_quantile_prediction_set.append((lower_bound, mu_on[i]+v_cut))
#                 elif mu_on[i]-v_cut>lower_bound and mu_on[i]+v_cut>=upper_bound:
#                     weighted_quantile_prediction_set.append((mu_on[i]-v_cut, upper_bound))
#                 else:
#                     weighted_quantile_prediction_set.append((lower_bound, upper_bound))

#         return weighted_quantile_prediction_set, cal_size_weighted

#     else:
#         raise ValueError(f"Unknown method: {method!r}")