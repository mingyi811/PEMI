# File: decision_permutation/selection.py
from turtle import ScrolledCanvas
from config import tau0, tau1, method
from typing import List
import numpy as np


def selection_rule_conformal_p_value(
    w_i: List[float],
    c_i: List[float],
    X_on: np.ndarray,
    Y_on: np.ndarray,
    mu_on: np.ndarray,
    perm_on: tuple,
    y_index: int,
    k: int,
    method: str = method
) -> bool:
    """
    Selection rule: fixed threshold or adaptive threshold.
    
    Parameters
    ----------
    w_i : List[float]
        weight of the current t.
    X_on : np.ndarray
        covariates of the calibration and test points.
    Y_on : np.ndarray
        outcomes of the calibration and test points.
    mu_on : np.ndarray
        mu of the calibration and test points.
    perm_on : np.ndarray
        permutation of the calibration and test points.
    y_index : int
        index of the current test point.
    method : str, optional
        "fixed threshold" or "adaptive threshold" .
    """


    if isinstance(mu_on, (np.float64, float, int)):
        return False


    if method == "fixed_threshold":
        # 实现 p-value 公式
        # p_{i,sigma}^{w,k} = (w_i + sum_{t!=j}^{i-1} w_t * 1{S_t >= S_i, Y_t <= c_t} + k * w_j * 1{S_j >= S_i}) / sum_{t=1}^i w_t
        
        #用当前的perm重新排序Y_on，scores
        scores=c_i[list(perm_on)]-mu_on[list(perm_on)]
        y_perm=Y_on[list(perm_on)]

            
        # 计算分子
        i=len(w_i)-1
        numerator = w_i[i]   # w_i
        
        # w_i + sum_{t!=j}^{i-1} w_t * 1{S_t >= S_i, Y_t <= c_t} + k * w_j * 1{S_j >= S_i}
        for t in range(i):
            if t==y_index:
                numerator += k*w_i[t]*(scores[t]>=scores[i])
            else:
                numerator += w_i[t]*(scores[t]>=scores[i] and y_perm[t]<=c_i[t])
        
        # 计算 p-value
        p_value = numerator / np.sum(w_i)
        
        # 与阈值比较
        threshold = 0.5
        return p_value > threshold
    
    elif method == "LORD_threshold":
        # 实现 p-value 公式
        # p_{i,sigma}^{w,k} = (w_i + sum_{t!=j}^{i-1} w_t * 1{S_t >= S_i, Y_t <= c_t} + k * w_j * 1{S_j >= S_i}) / sum_{t=1}^i w_t
        
        #设置想要的fdr control
        alpha_fdr = 0.4
        #用当前的perm重新排序Y_on，scores
        scores=c_i[list(perm_on)]-mu_on[list(perm_on)]
        y_perm=Y_on[list(perm_on)]

            
        # 计算分子
        i=len(w_i)-1
        numerator = w_i[i]   # w_i
        # 计算lord threshold
        w0 = 0.3*alpha_fdr   # Initial weight
        b0 = 0.4*w0  # Penalty for rejections
        pvalues=[]
        taus=[]

        #计算所有的gamma_t
        # Compute gamma_t = log(log(max(t, 2))) / (t * exp(sqrt(log(t))))
        from math import log, exp, sqrt
        gammas=[]
        #这个是paper中推荐的lord的gamma_t
        # for t in range(i+1):
        #     t_max = max(t+1, 2)
        #     log_t = log(t_max) 
        #     gamma_t = log_t / ((t+1) * exp(sqrt(log(t+1))))
        #     gammas.append(gamma_t)
        
        #试一个别的
        for t in range(i+1):
            gamma_t=1.0/(t+1)**1.6
            gammas.append(gamma_t)


        # 计算所有时刻的pvalues
        # w_i + sum_{t!=j}^{i-1} w_t * 1{S_t >= S_i, Y_t <= c_t} + k * w_j * 1{S_j >= S_i}
        for j in range(i+1):
            for t in range(j):
                numerator=w_i[j]
                if t==y_index:
                    numerator += k*w_i[t]*(scores[t]>=scores[j])
                else:
                    numerator += w_i[t]*(scores[t]>=scores[j] and y_perm[t]<=c_i[t])
            # 计算 p-value
            p_value = numerator / np.sum(w_i)
            pvalues.append(p_value)

            alpha = compute_alpha_lord(j, taus, gammas, w0, b0)
            if p_value <= alpha:
                taus.append(j)
        
        #计算所有时刻的alpha和tau
        #taus=[] 
        #for j in range(i+1):
        #    alpha = compute_alpha(j, taus, gammas, w0, b0)
        #    if pvalues[j] <= alpha:
        #        taus.append(j)

        adaptive_threshold = compute_alpha_lord(i, taus, gammas, w0, b0)
        return pvalues[i] > adaptive_threshold
    
    elif method == "E_LOND_threshold":
        # 实现 p-value 公式
        # p_{i,sigma}^{w,k} = (w_i + sum_{t!=j}^{i-1} w_t * 1{S_t >= S_i, Y_t <= c_t} + k * w_j * 1{S_j >= S_i}) / sum_{t=1}^i w_t
        
        #设置想要的fdr control
        alpha_fdr = 0.4
        #用当前的perm重新排序Y_on，scores
        scores=c_i[list(perm_on)]-mu_on[list(perm_on)]
        y_perm=Y_on[list(perm_on)]

            
        # 计算分子
        i=len(w_i)-1
        numerator_neg=0
        numerator_pos=0
        #numerator = w_i[i]   # w_i
        # 计算lord threshold
        #w0 = 0.3*alpha_fdr   # Initial weight
        #b0 = 0.4*w0  # Penalty for rejections
        pvalues_neg=[]
        pvalues_pos=[]
        taus_neg=[]
        taus_pos=[]

        #计算所有的gamma_t
        # Compute gamma_t = log(log(max(t, 2))) / (t * exp(sqrt(log(t))))
        from math import log, exp, sqrt
        gammas=[]
        #这个是paper中推荐的lord的gamma_t
        # for t in range(i+1):
        #     t_max = max(t+1, 2)
        #     log_t = log(t_max) 
        #     gamma_t = log_t / ((t+1) * exp(sqrt(log(t+1))))
        #     gammas.append(gamma_t)
        
        #试一个别的
        for t in range(i+1):
            gamma_t=1.0/(t+1)**1.6
            gammas.append(gamma_t)


        # 计算所有时刻的pvalues
        # w_i + sum_{t!=j}^{i-1} w_t * 1{S_t >= S_i, Y_t <= c_t} + k * w_j * 1{S_j >= S_i}
        for j in range(i+1):
            for t in range(j):
                numerator_neg=0
                numerator_pos=w_i[j]
                if t==y_index:
                    numerator_neg += k*w_i[t]*(scores[t]>=scores[j])
                    numerator_pos += k*w_i[t]*(scores[t]>=scores[j])
                else:
                    numerator_neg += w_i[t]*(scores[t]>=scores[j] and y_perm[t]<=c_i[t])
                    numerator_pos += w_i[t]*(scores[t]>=scores[j] and y_perm[t]<=c_i[t])
            # 计算 p-value
            p_value_neg = numerator_neg / np.sum(w_i)
            p_value_pos = numerator_pos / np.sum(w_i)
            pvalues_neg.append(p_value_neg)
            pvalues_pos.append(p_value_pos)

            alpha_neg = compute_alpha_lond(j, taus_neg, gammas, alpha_fdr)
            alpha_pos = compute_alpha_lond(j, taus_pos, gammas, alpha_fdr)
            if p_value_neg <= alpha_neg:
                taus_neg.append(j)
            if p_value_pos <= alpha_pos:
                taus_pos.append(j)
        #计算所有时刻的alpha和tau
        #taus=[] 
        #for j in range(i+1):
        #    alpha = compute_alpha_lond(j, taus, gammas, w0, b0)
        #    if pvalues[j] <= alpha:
        #        taus.append(j)

        #adaptive_threshold = compute_alpha(i, taus, gammas, w0, b0)
        #return pvalues[i] > adaptive_threshold

        #计算最后两个alpha
        alpha_neg_i = compute_alpha_lond(i, taus_neg, gammas, alpha_fdr)
        alpha_pos_i = compute_alpha_lond(i, taus_pos, gammas, alpha_fdr)
        #计算evalue
        e_lond=(pvalues_pos[i]<=alpha_pos_i)/alpha_neg_i

        return e_lond<1/alpha_pos_i

    else:
        raise ValueError(f"Unknown method: {method!r}")
    return 0

def compute_alpha_lord(t:int, taus: List[int], gammas: List[float], w0: float, b0: float):
    adaptive_threshold = w0 * gammas[t-1]
    # Simulate rejection times (tau_j), here we assume a list of past rejection times
    for tau_j in taus:
        if tau_j != 0:  # t1 is the first rejection time, adjust accordingly
            adaptive_threshold += gammas[t-tau_j] * b0
    return adaptive_threshold

def compute_alpha_lond(t:int, taus: List[int], gammas: List[float], alpha_fdr: float):
    taus_length=len(taus)
    alpha_lond=alpha_fdr*gammas[t-1]*(taus_length+1)
    return alpha_lond
