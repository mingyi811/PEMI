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
    
    elif method == "weighted_quantile":
        values = np.array(mu_on)
        n = len(values)
        distances = np.arange(n, 0, -1)
        weights = distances / distances.sum()

        sorter = np.argsort(values)
        sorted_vals = values[sorter]
        sorted_weights = weights[sorter]

        cum_weights = np.cumsum(sorted_weights)
        idx = np.searchsorted(cum_weights, 0.7, side="left")
        threshold = sorted_vals[min(idx, n - 1)]

    else:
        raise ValueError(f"Unknown method: {method!r}")
    return 0

