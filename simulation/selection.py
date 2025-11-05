from config import method
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

    q = 0.9  

    if method == "quantile":
        # 70% quantile
        threshold = np.quantile(mu_history, q)

    elif method == "weighted_quantile":
        values = np.array(mu_history)
        n = len(values)
        
        distances = np.arange(n, 0, -1)
        # The closer to the test point, the greater the weight, raw_weights is exponentially decaying
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
        # The closer to the test point, the greater the weight, raw_weights is exponentially decaying
        raw_weights = 0.5 ** distances
        weights = raw_weights / raw_weights.sum()
        threshold = np.sum(values * weights) / np.sum(weights)
    elif method == "former point":
        threshold = mu_history[-1]
    elif method == "max":
        threshold = max(mu_history)
        if mu_history[-1] == threshold:
            return True
        else:
            return False
    else:
        raise ValueError(f"Unknown method: {method!r}")

    return mu_t > threshold
