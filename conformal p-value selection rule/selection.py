
from turtle import ScrolledCanvas
from config import method
from typing import List
import numpy as np


def selection_rule_conformal_p_value(
    w_i: List[float],
    c_i: List[float],
    Y_on: np.ndarray,
    mu_on: np.ndarray,
    Y_cal: np.ndarray,
    mu_cal: np.ndarray,
    c_cal: np.ndarray,
    w_i_cal: np.ndarray,
    perm_on: tuple,
    y_index: int,
    k: int,
    u: float,
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
    X_cal : np.ndarray
        covariates of the calibration points.
    Y_cal : np.ndarray
        outcomes of the calibration points.
    mu_cal : np.ndarray
        mu of the calibration points.
    c_cal : np.ndarray
        c of the calibration points.
    w_i_cal : np.ndarray
        weight of the calibration points.
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
        # p_{i,sigma}^{w,k} = (w_i + sum_{t!=j}^{i-1} w_t * 1{S_t >= S_i, Y_t <= c_t} + k * w_j * 1{S_j >= S_i}) / sum_{t=1}^i w_t
        
        #Reorder Y_on and scores using the current perm
        scores=mu_on[list(perm_on)]-c_i[list(perm_on)]
        y_perm=Y_on[list(perm_on)]

            
        # Calculate the numerator
        i=len(w_i)-1
        numerator = w_i[i]   # w_i
        
        # w_i + sum_{t!=j}^{i-1} w_t * 1{S_t >= S_i, Y_t <= c_t} + k * w_j * 1{S_j >= S_i}
        for t in range(i):
            if t==y_index:
                numerator += k*w_i[t]*(scores[t]>=scores[i])
            else:
                numerator += w_i[t]*(scores[t]>=scores[i] and y_perm[t]<=c_i[t])
        
        # Calculate the p-value
        p_value = numerator / np.sum(w_i)
        
        # Compare with the threshold
        threshold = 0.3
        return p_value <= threshold
    
    elif method == "LOND_threshold":
        # p_{i,sigma}^{w,k} = (w_i + sum_{t!=j}^{i-1} w_t * 1{S_t >= S_i, Y_t <= c_t} + k * w_j * 1{S_j >= S_i}) / sum_{t=1}^i w_t
        
        #Set the desired fdr control
        alpha_fdr = 0.4
        #Reorder Y_on and scores using the current perm
        scores=mu_on[list(perm_on)]-c_i[list(perm_on)]
        y_perm=Y_on[list(perm_on)]

            
        # Calculate the numerator
        i=len(w_i)-1
        numerator = w_i[i]   # w_i
    
        pvalues=[]
        taus=[]

        # Calculate the gamma_t
        import mpmath as mp
        p=1.6
        c=1.0/float(mp.zeta(p))
        gammas=[c * (t+1)**(-p) for t in range(i+1)]


        # Calculate all the pvalues
        # w_i + sum_{t!=j}^{i-1} w_t * 1{S_t >= S_i, Y_t <= c_t} + k * w_j * 1{S_j >= S_i}
        for j in range(i+1):
            numerator=w_i[j]
            for t in range(j):
                if t==y_index:
                    numerator += k*w_i[t]*(scores[t]>=scores[j])
                else:
                    numerator += w_i[t]*(scores[t]>=scores[j] and y_perm[t]<=c_i[t])
            # Calculate the p-value
            p_value = numerator / np.sum(w_i[:j+1])
            pvalues.append(p_value)

            alpha = compute_alpha_lond(j, taus, gammas, alpha_fdr)
            if p_value <= alpha:
                taus.append(j)
        
        return pvalues[i] <= alpha
    
    elif method == "online_BH_threshold":
        # p_{i,sigma}^{w,k} = (w_i + sum_{t!=j}^{i-1} w_t * 1{S_t >= S_i, Y_t <= c_t} + k * w_j * 1{S_j >= S_i}) / sum_{t=1}^i w_t
        
        #Set the desired fdr control
        alpha_fdr = 0.4
        #Reorder Y_on and scores using the current perm
        scores=mu_on[list(perm_on)]-c_i[list(perm_on)]
        y_perm=Y_on[list(perm_on)]

            
        # Calculate the numerator
        i=len(w_i)-1
        numerator = w_i[i]   # w_i
        pvalues=[]
        taus=[]

        # Calculate the gamma_t
        import mpmath as mp
        p=1.6
        c=1.0/float(mp.zeta(p))
        gammas=[c * (t+1)**(-p) for t in range(i+1)]

        # Calculate all the pvalues
        # w_i + sum_{t!=j}^{i-1} w_t * 1{S_t >= S_i, Y_t <= c_t} + k * w_j * 1{S_j >= S_i}
        for j in range(i+1):
            numerator=w_i[j]*u
            for t in range(j):
                if t==y_index:
                    numerator += k*w_i[t]*(scores[t]>=scores[j])
                else:
                    numerator += w_i[t]*(scores[t]>=scores[j] and y_perm[t]<=c_i[t])
            # Calculate the p-value
            p_value = numerator / np.sum(w_i[:j+1])
            pvalues.append(p_value)

            alpha = compute_alpha_online_BH(j, pvalues, gammas, alpha_fdr)
            if p_value <= alpha:
                taus.append(j)
        
        return pvalues[i] <= alpha
    
    elif method == "E_LOND_threshold":
        # p_{i,sigma}^{w,k} = (w_i + sum_{t!=j}^{i-1} w_t * 1{S_t >= S_i, Y_t <= c_t} + k * w_j * 1{S_j >= S_i}) / sum_{t=1}^i w_t
        
        #Set the desired fdr control
        alpha_fdr = 0.5
        #Reorder Y_on and scores using the current perm
        scores=mu_on[list(perm_on)]-c_i[list(perm_on)]
        y_perm=Y_on[list(perm_on)]

        n_cal=len(mu_cal)
        scores_cal=mu_cal-c_cal

            
        # Calculate the numerator
        i=len(w_i)-1
        numerator_neg=0
        numerator_pos=0
        numerator_j=0

        pvalues_neg=[]
        pvalues_pos=[]
        pvalues_j=[]
        taus_neg=[]
        taus_pos=[]
        taus_e_lond=[]

        from math import log, exp, sqrt
        gammas=[]
        
        # Calculate the gamma_t
        import mpmath as mp
        p=1.6
        c=1.0/float(mp.zeta(p))
        gammas=[c * (t+1)**(-p) for t in range(i+1)]


        # Calculate all the pvalues
        # w_i + sum_{t!=j}^{i-1} w_t * 1{S_t >= S_i, Y_t <= c_t} + k * w_j * 1{S_j >= S_i}
        for j in range(i):
            numerator_neg=0
            numerator_pos=u
            numerator_j=u
            for t in range(n_cal):
                if j==y_index+n_cal:
                    numerator_neg += k*(scores_cal[t]>=scores[j])
                    numerator_pos += k*(scores_cal[t]>=scores[j])
                    numerator_j += k*(scores_cal[t]>=scores[j])
                else:
                    numerator_neg += (scores_cal[t]>=scores[j] and Y_cal[t]<=c_cal[t])
                    numerator_pos += (scores_cal[t]>=scores[j] and Y_cal[t]<=c_cal[t])
                    numerator_j += (scores_cal[t]>=scores[j] and Y_cal[t]<=c_cal[t])
            # Calculate the p-value
            p_value_neg = numerator_neg / (n_cal+1)
            p_value_pos = numerator_pos / (n_cal+1)
            p_value_j = numerator_j / (n_cal+1)
            pvalues_neg.append(p_value_neg)
            pvalues_pos.append(p_value_pos)
            pvalues_j.append(p_value_j)

            alpha_neg = compute_alpha_lond(j, taus_neg, gammas, alpha_fdr, 1)
            alpha_pos = compute_alpha_lond(j, taus_pos, gammas, alpha_fdr, 1)
            alpha_j = compute_alpha_lond(j, taus_e_lond, gammas, alpha_fdr, u)

            e_value_j = (p_value_j<=alpha_pos)/alpha_neg

            if p_value_neg <= alpha_neg:
                taus_neg.append(j)
            if p_value_pos <= alpha_pos:
                taus_pos.append(j)
            if e_value_j >= 1/alpha_j:
                taus_e_lond.append(j)

        # Calculate the p-value_i
        numerator_i_final=w_i[i]*u
        for t in range(n_cal):
            numerator_i_final+=(scores_cal[t]>=scores[i] and Y_cal[t]<=c_cal[t])
        p_i=numerator_i_final/(n_cal+1)

        # Calculate the last two alpha
        alpha_neg_i = compute_alpha_lond(i, taus_neg, gammas, alpha_fdr, 1)
        alpha_pos_i = compute_alpha_lond(i, taus_pos, gammas, alpha_fdr, 1)
        alpha_i = compute_alpha_lond(i, taus_e_lond, gammas, alpha_fdr, u)
        # Calculate the e-value
        e_lond=(p_i<=alpha_pos_i)/alpha_neg_i
        is_selected=(e_lond >= 1/alpha_i)

        return is_selected

    else:
        raise ValueError(f"Unknown method: {method!r}")


def compute_alpha_lond(t:int, taus: List[int], gammas: List[float], alpha_fdr: float, u: float):
    taus_length=len(taus)
    alpha_lond=alpha_fdr * gammas[t] * (taus_length+1)
    return alpha_lond/u

def compute_alpha_online_BH(t: int, pvalues: List[float], gammas: List[float], alpha_fdr: float):
    k_star = 0
    for k in range(1, t + 1):  
        sum_val = sum(1 for j in range(t) if pvalues[j] <= k * alpha_fdr * gammas[j])
        if sum_val < k:
            k_star = k - 1
            break
    else:
        k_star = t  
    threshold = k_star * alpha_fdr * gammas[t-1] if k_star > 0 else 0
    return threshold

