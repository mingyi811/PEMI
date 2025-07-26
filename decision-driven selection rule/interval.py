# File: decision_permutation/interval.py
import numpy as np
import math
from typing import Tuple, List, Callable
from config import j_feature
from selection import selection_rule


def get_smoothed_cutoff_with_N1(scores: List[float], alpha: float, N1: int) -> float:
    """
    Compute p-value for a given v_test using randomized smoothing.
    p(y) = (#{scores > v_test} + U * (#{scores == v_test} + N1)) / (len(scores) + N1)
    """
    #方案1: upper quantile的cutoff(理论上最严格，valid但会很大)
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
    # scores_arr = np.array(scores)
    # total = len(scores_arr) + N1
    # U=np.random.uniform(0,1)
    # v_upper = math.inf
    # for v in np.sort(np.unique(scores_arr))[::-1]:
    #     num_gt = np.sum(scores_arr > v)
    #     num_eq = np.sum(scores_arr == v)
    #     p_val = (num_gt + U * (num_eq + N1)) / (total)
    #     if p_val >= alpha:
    #         return v_upper
    #     v_upper = v
    # return v_upper

    #方案3: 自己计算的expectation=1-alpha的cutoff(算出来的expectation等于1-alpha，但实际做出来会有波动)
    # scores_arr = np.array(scores)
    # total = len(scores_arr) + N1
    # U=np.random.uniform(0,1) 
    # v_upper = math.inf
    # num_gt_upper = 0
    # num_eq_upper = 0    
    # v_list = np.sort(np.unique(scores_arr))[::-1]
    # for v in v_list:
    #     num_gt = np.sum(scores_arr > v)
    #     num_eq = np.sum(scores_arr == v)
    #     prob = (alpha*total-0.5*N1-num_gt_upper-0.5*num_eq_upper)/(num_gt+0.5*num_eq-num_gt_upper-0.5*num_eq_upper)
    #     p_val = (num_gt + U * (num_eq + N1)) / (total)
    #     if p_val >= alpha:
    #         if U < prob:
    #             return v
    #         else:
    #             return v_upper
    #     v_upper = v
    #     num_gt_upper = num_gt
    #     num_eq_upper = num_eq
    # return math.inf

    #方案4: 检测一下是否方案二是因为pvalues not monotone的问题（结果来看好像也不是。。。那可能是这个pvalue本身就not valid）
    # scores_arr = np.array(scores)
    # pvalues_and_v_tuples = []
    # total = len(scores_arr) + N1
    # U=np.random.uniform(0,1)
    # for v in np.sort(np.unique(scores_arr))[::-1]:
    #     num_gt = np.sum(scores_arr > v)
    #     num_eq = np.sum(scores_arr == v)
    #     p_val = (num_gt + U * (num_eq + N1)) / (total)
    #     pvalues_and_v_tuples.append((p_val, v))
    # # 假设 pvalues_and_v_tuples 是 [(pval, v), ...]
    # pvalues_and_v_tuples_sorted = sorted(pvalues_and_v_tuples, key=lambda x: x[0])
    # for i, (pval, v) in enumerate(pvalues_and_v_tuples_sorted):
    #     if pval >= alpha:
    #         if i == 0:
    #             # 如果第一个pvalue就大于等于alpha，返回第一个v
    #             return v
    #         else:
    #             return pvalues_and_v_tuples_sorted[i-1][1]
    # # 如果都没有大于等于alpha的pvalue，返回最后一个v
    # return math.inf


def construct_prediction_interval(
    t: int,
    X_on: np.ndarray,
    Y_on: np.ndarray,
    mu_on: np.ndarray,
    M: int,
    alpha: float
) -> Tuple[float, float, int]:
    """
    Build prediction interval at time t via randomized permutations.
    """
    if t == 0:
        return -math.inf, math.inf, 0
    i = t
    indices = list(range(i + 1))
    base_perm = tuple(indices) #base_perm是observed data的permutation

    perms_list = [base_perm]
    max_perms = min(M + 1, math.factorial(i + 1))
    #perms_list是所有最开始random permutation的list
    while len(perms_list) < max_perms:
        pi = tuple(np.random.permutation(indices))
        if pi not in perms_list:
            perms_list.append(pi)

    perms_selected = []
    X_slice = X_on[: i + 1] #X_slice是observed data的X
    for pi in perms_list:
        cum = 0
        for k in range(i):
            if selection_rule(X_slice[pi[k], j_feature], cum):
                cum += 1
        if selection_rule(X_slice[pi[i], j_feature], cum):
            perms_selected.append(pi)
    cal_size = len(perms_selected)

    # Separate observed vs. visible residuals
    N1 = 0
    scores: List[float] = []
    for pi in perms_selected:
        if pi[i] == i:
            N1 += 1 #N1是我们不知道具体score但知道这个permutation的score是等于observed data的score的
        else:
            k = pi[i]
            scores.append(float(np.abs(Y_on[k] - mu_on[k]))) #最后一个点的residual作为score

    if not scores:
        return -math.inf, math.inf, 0

    scores.append(float(np.inf)) #scores中append inf

    #U=np.random.uniform(0,1)
    v_cut = get_smoothed_cutoff_with_N1(scores, alpha, N1)
    return mu_on[t] - v_cut, mu_on[t] + v_cut, cal_size