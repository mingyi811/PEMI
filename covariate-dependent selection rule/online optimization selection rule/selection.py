from config import method
from typing import List
import numpy as np


def selection_rule_optimization(
    mu_1: List[float],
    mu_2: List[float],
    mu_3: List[float],
    method: str = method
) -> bool:
    """
    Selection rule: online optimization.
    
    Parameters
    ----------
    mu_1 : List[float]
        mu_1 of the current test point.
    mu_2 : List[float]
        mu_2 of the current test point.
    mu_3 : List[float]
        mu_3 of the current test point.
    method : str, optional
        "optimization".
    """
    if len(mu_1) == 0:
        return False

    if method == "direct":
        eta = 0.01  
        gamma = 0.4  
        tau = 0.0  

        phi = False
        for i in range(len(mu_1)):
            mus = [mu_1[i], mu_2[i], mu_3[i]]
            s_t = np.var(mus)
            phi = s_t >= tau
            tau += eta * (int(phi) - gamma)  
        return phi
    
    elif method == "optimization":
        gamma = 0.4  

        past_s = []
        for i in range(len(mu_1) - 1):
            mus = [mu_1[i], mu_2[i], mu_3[i]]
            past_s.append(np.var(mus))

        past_len = len(past_s)
        if past_len == 0:
            return False

        # Sort past_s in descending order
        s_sorted = sorted(past_s, reverse=True)

        # Initialize
        i = 0
        cum_num = 0
        tau = np.inf  # If no selection, high threshold

        while i < past_len:
            current_s = s_sorted[i]
            group_size = 0
            while i < past_len and s_sorted[i] == current_s:
                group_size += 1
                i += 1

            if (cum_num + group_size) / past_len <= gamma:
                cum_num += group_size
                tau = current_s  # Update tau to include this group
            else:
                break

        # Compute s_t for current
        current_i = len(mu_1) - 1
        mus_current = [mu_1[current_i], mu_2[current_i], mu_3[current_i]]
        s_t = np.var(mus_current)

        if tau == np.inf:
            return False
        else:
            return s_t >= tau
    
    

    else:
        raise ValueError(f"Unknown method: {method!r}")


