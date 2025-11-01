# File: decision_permutation/selection.py
from config import q, method
from typing import List
import numpy as np


def selection_rule_earlier_outcomes(
    mu_t: float,
    y_history: List[float],
    method: str = method,
    q: float = q
) -> bool:
    """
    Selection rule: quantile or weighted quantile.
    
    Parameters
    ----------
    mu_t : float
        mu of the current test point.
    y_history : List[float]
        mu of the past test points.
    method : str, optional
        "quantile" or "weighted quantile".
    """
    if len(y_history) == 0:
        return False

    if method == "quantile":
        # 70% quantile
        threshold = np.quantile(y_history, q, interpolation="higher")

    elif method == "weighted_quantile":
        values = np.array(y_history)
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
    
    else:
        raise ValueError(f"Unknown method: {method!r}")

    return mu_t > threshold
