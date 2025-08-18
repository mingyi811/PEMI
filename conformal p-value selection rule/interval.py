# File: decision_permutation/interval.py
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

# def construct_prediction_interval(
#     t: int,
#     X_on: np.ndarray,
#     Y_on: np.ndarray,
#     mu_on: np.ndarray,
#     c_on: np.ndarray,
#     w_i: np.ndarray,
#     M: int,
#     alpha: float,
#     quantile_method: str = "upper"
# ) -> Tuple[float, float, float, float, int]:
#     """
#     Build symmetric prediction interval at time t.
#     This function works with selection_rule of form (X_j_val, cum_selected) -> bool
#     or (mu_t, mu_history) -> bool, depending on the rule logic.
#     """
#     if t == 0:
#         return -math.inf, math.inf, -math.inf, math.inf, 0
#     i = t
#     indices = list(range(i + 1))
#     base_perm = tuple(indices) #base_perm是observed data的permutation

#     perms_list = [base_perm]
#     max_perms = min(M + 1, math.factorial(i + 1))
#     #perms_list是所有最开始random permutation的list

#     #方案1:直接用最简单的random permutation
#     while len(perms_list) < max_perms:
#         pi = tuple(np.random.permutation(indices))
#         if pi not in perms_list:
#             perms_list.append(pi)
    


#     #方案2:我们试试分层抽样
#     # perms_per_last = max_perms // (t + 1)  # 每个最后一个位置值的排列数
#     # extra_perms = max_perms % (t + 1)  # 余数，分配到前 extra_perms 个层

#     # for last_val in range(t + 1):
#     #     # 确定该层的排列数
#     #     n_perms = perms_per_last + (1 if last_val < extra_perms else 0)
        
#     #     # 如果 last_val 是 base_perm 的最后一个值，base_perm 已包含，减少一次
#     #     if last_val == base_perm[-1] and n_perms > 0:
#     #         n_perms -= 1
        
#     #     # 生成排列
#     #     remaining_indices = [x for x in range(t + 1) if x != last_val]  # 除去 last_val 的索引
#     #     for _ in range(n_perms):
#     #         # 对剩余 t 个位置随机排列
#     #         perm = list(np.random.permutation(remaining_indices))
#     #         # 添加最后一个位置
#     #         perm.append(last_val)
#     #         perms_list.append(tuple(perm))

#     #initialize w_i
#     #w_i = np.random.uniform(0,1,len(mu_on))
    
#     #对于y_t的大小分类讨论，分别讨论y_t>c_t和y_t<c_t的情况
#     perms_selected_k_0 = []
#     perms_selected_k_1 = []
#     #suppose y_t>c_t
#     for k in [0,1]:
#         if k==0:
#             for pi in perms_list:
#                 #对于y_t>c_t的情况，k=0
#                 y_index=pi.index(t)

#                 if selection_rule_conformal_p_value(w_i, c_on, X_on, Y_on, mu_on, pi, y_index, k,  method=method):
#                     perms_selected_k_0.append(pi)
#         else:
#             for pi in perms_list:
#                 #对于y_t<c_t的情况，k=1
#                 y_index=pi.index(t)
#                 if selection_rule_conformal_p_value(w_i, c_on, X_on, Y_on, mu_on, pi, y_index, k,  method=method):
#                     perms_selected_k_1.append(pi)
#     cal_size = (len(perms_selected_k_0) + len(perms_selected_k_1))/2



#     N1 = 0
#     scores_k_0 = []
#     scores_k_1 = []
#     for pi in perms_selected_k_0:
#         if pi[i] == i:
#             N1 += 1 #N1是我们不知道具体score但知道这个permutation的score是等于observed data的score的
#         else:
#             k = pi[i]
#             scores_k_0.append(abs(Y_on[k] - mu_on[k])) #最后一个点的residual作为score
#     for pi in perms_selected_k_1:
#         if pi[i] == i:
#             N1 += 1 #N1是我们不知道具体score但知道这个permutation的score是等于observed data的score的
#         else:
#             k = pi[i]
#             scores_k_1.append(abs(Y_on[k] - mu_on[k])) #最后一个点的residual作为score

#     if not scores_k_0:
#         return -math.inf, math.inf, -math.inf, math.inf, 0
#     scores_k_0.append(math.inf) #scores中append inf
#     if not scores_k_1:
#         return -math.inf, math.inf, -math.inf, math.inf, 0
#     scores_k_1.append(math.inf) #scores中append inf

#     v_cut_k_0 = get_smoothed_cutoff_with_N1(scores_k_0, alpha, N1, quantile_method=quantile_method)
#     v_cut_k_1 = get_smoothed_cutoff_with_N1(scores_k_1, alpha, N1, quantile_method=quantile_method)

#     if mu_on[t]-v_cut_k_1>c_on[t]:
#         bound_1=0
#         bound_2=0
#     else:
#         bound_1=mu_on[t]-v_cut_k_1
#         if mu_on[t]+v_cut_k_1<c_on[t]:
#             bound_2=mu_on[t]+v_cut_k_1
#         else:
#             bound_2=c_on[t]
#     if mu_on[t]+v_cut_k_0<c_on[t]:
#         bound_3=0
#         bound_4=0
#     else:
#         bound_4=mu_on[t]+v_cut_k_0
#         if mu_on[t]-v_cut_k_0>c_on[t]:
#             bound_3=mu_on[t]-v_cut_k_0
#         else:
#             bound_3=c_on[t]

#     return bound_1, bound_2, bound_3, bound_4, cal_size



def construct_prediction_interval(
    t: int,
    X_on: np.ndarray,
    Y_on: np.ndarray,
    mu_on: np.ndarray,
    X_off: np.ndarray,
    Y_off: np.ndarray,
    mu_off: np.ndarray,
    c_on: np.ndarray,
    w_i_on: np.ndarray,
    c_off: np.ndarray,
    w_i_off: np.ndarray,
    M: int,
    alpha: float,
    quantile_method: str = "upper"
) -> Tuple[float, float, float, float, int]:
    """
    Build symmetric prediction interval at time t.
    This function works with selection_rule of form (X_j_val, cum_selected) -> bool
    or (mu_t, mu_history) -> bool, depending on the rule logic.
    """
    n_off = len(X_off)
    X_aug = np.concatenate((X_off,X_on))
    mu_aug = np.concatenate((mu_off,mu_on))
    Y_aug = np.concatenate((Y_off,Y_on))
    c_aug = np.concatenate((c_off,c_on))
    w_i_aug = np.concatenate((w_i_off,w_i_on))
    if t+n_off == 0:
        return -math.inf, math.inf, -math.inf, math.inf, 0
    i = t
    indices_aug = list(range(i + n_off + 1))
    base_perm_aug = tuple(indices_aug) #base_perm是observed data的permutation

    perms_list_aug = [base_perm_aug]
    max_perms_aug = min(M + 1, math.factorial(i + 1 + n_off))
    #perms_list是所有最开始random permutation的list

    #方案1:直接用最简单的random permutation
    while len(perms_list_aug) < max_perms_aug:
        pi = tuple(np.random.permutation(indices_aug))
        if pi not in perms_list_aug:
            perms_list_aug.append(pi)
    


    #方案2:我们试试分层抽样
    # perms_per_last = max_perms // (t + 1)  # 每个最后一个位置值的排列数
    # extra_perms = max_perms % (t + 1)  # 余数，分配到前 extra_perms 个层

    # for last_val in range(t + 1):
    #     # 确定该层的排列数
    #     n_perms = perms_per_last + (1 if last_val < extra_perms else 0)
        
    #     # 如果 last_val 是 base_perm 的最后一个值，base_perm 已包含，减少一次
    #     if last_val == base_perm[-1] and n_perms > 0:
    #         n_perms -= 1
        
    #     # 生成排列
    #     remaining_indices = [x for x in range(t + 1) if x != last_val]  # 除去 last_val 的索引
    #     for _ in range(n_perms):
    #         # 对剩余 t 个位置随机排列
    #         perm = list(np.random.permutation(remaining_indices))
    #         # 添加最后一个位置
    #         perm.append(last_val)
    #         perms_list.append(tuple(perm))

    #initialize w_i
    #w_i = np.random.uniform(0,1,len(mu_on))
    
    #对于y_t的大小分类讨论，分别讨论y_t>c_t和y_t<c_t的情况
    perms_selected_k_0 = []
    perms_selected_k_1 = []
    #suppose y_t>c_t
    for k in [0,1]:
        if k==0:
            for pi in perms_list_aug:
                #对于y_t>c_t的情况，k=0
                y_index_aug=pi.index(t+n_off)
                if y_index_aug>=n_off:
                    y_index=y_index_aug-n_off
                else:
                    y_index=math.inf
                mu_select = mu_aug[list(pi)][n_off:i+n_off+1]  # 从 n_off 到 i-1，去掉 offline data
                w_i_select = w_i_aug[list(pi)][n_off:i+n_off+1]
                c_select = c_aug[list(pi)][n_off:i+n_off+1]
                X_select = X_aug[list(pi)][n_off:i+n_off+1]
                Y_select = Y_aug[list(pi)][n_off:i+n_off+1]
                pi_select = tuple(list(range(i + 1)))

                if selection_rule_conformal_p_value(w_i_select, c_select, X_select, Y_select, mu_select, pi_select, y_index, k,  method=method):
                    perms_selected_k_0.append(pi)
        else:
            for pi in perms_list_aug:
                #对于y_t<c_t的情况，k=1
                y_index_aug=pi.index(t+n_off)
                if y_index_aug>=n_off:
                    y_index=y_index_aug-n_off
                else:
                    y_index=math.inf
                mu_select = mu_aug[list(pi)][n_off:i+n_off+1]  # 从 n_off 到 i-1，去掉 offline data
                w_i_select = w_i_aug[list(pi)][n_off:i+n_off+1]
                c_select = c_aug[list(pi)][n_off:i+n_off+1]
                X_select = X_aug[list(pi)][n_off:i+n_off+1]
                Y_select = Y_aug[list(pi)][n_off:i+n_off+1]
                pi_select = tuple(list(range(i + 1)))

                if selection_rule_conformal_p_value(w_i_select, c_select, X_select, Y_select, mu_select, pi_select, y_index, k,  method=method):
                    perms_selected_k_1.append(pi)
    cal_size = (len(perms_selected_k_0) + len(perms_selected_k_1))/2



    N1 = 0
    scores_k_0 = []
    scores_k_1 = []
    for pi in perms_selected_k_0:
        if pi[i+n_off] == i+n_off:
            N1 += 1 #N1是我们不知道具体score但知道这个permutation的score是等于observed data的score的
        else:
            k = pi[i+n_off]
            scores_k_0.append(abs(Y_aug[k] - mu_aug[k])) #最后一个点的residual作为score
    for pi in perms_selected_k_1:
        if pi[i+n_off] == i+n_off:
            N1 += 1 #N1是我们不知道具体score但知道这个permutation的score是等于observed data的score的
        else:
            k = pi[i+n_off]
            scores_k_1.append(abs(Y_aug[k] - mu_aug[k])) #最后一个点的residual作为score

    if not scores_k_0:
        return -math.inf, math.inf, -math.inf, math.inf, 0
    scores_k_0.append(math.inf) #scores中append inf
    if not scores_k_1:
        return -math.inf, math.inf, -math.inf, math.inf, 0
    scores_k_1.append(math.inf) #scores中append inf

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
