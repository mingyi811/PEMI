# File: decision_permutation/selection.py
from config import tau0, tau1, method
from typing import List
import numpy as np


def selection_rule_covariate_dependent(
    mu_t: float,
    mu_history: List[float],
    method: str = method
) -> bool:
    """
    Selection rule: quantile or weighted quantile or weighted average.
    
    Parameters
    ----------
    mu_t : float
        mu of the current test point.
    mu_history : List[float]
        mu of the past test points.
    method : str, optional
        "quantile" or "weighted quantile" or "weighted average".
    """
    if len(mu_history) == 0:
        return False

    q = 0.7  # 分位点

    if method == "quantile":
        # 70% quantile
        threshold = np.quantile(mu_history, q)

    elif method == "weighted_quantile":
        values = np.array(mu_history)
        n = len(values)
        distances = np.arange(n, 0, -1)
        # 越靠近测试点权重越大，raw_weights是指数衰减的
        raw_weights = 0.5 ** distances
        weights = raw_weights / raw_weights.sum()

        sorter = np.argsort(values)
        sorted_vals = values[sorter]
        sorted_weights = weights[sorter]

        cum_weights = np.cumsum(sorted_weights)
        idx = np.searchsorted(cum_weights, q, side="left")
        threshold = sorted_vals[min(idx, n - 1)]
    
    elif method == "weighted_average":
        values = np.array(mu_history)
        n = len(values)
        distances = np.arange(n, 0, -1)
        # 越靠近测试点权重越大，raw_weights是指数衰减的
        raw_weights = 0.5 ** distances
        weights = raw_weights / raw_weights.sum()
        threshold = np.sum(values * weights) / np.sum(weights)

    else:
        raise ValueError(f"Unknown method: {method!r}")

    return mu_t > threshold
