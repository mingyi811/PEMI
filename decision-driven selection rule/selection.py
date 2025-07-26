# File: decision_permutation/selection.py
from config import tau0, tau1


def selection_rule(X_j_val: float, cum_selected: int) -> bool:
    return X_j_val < (tau1 + cum_selected / tau0)

