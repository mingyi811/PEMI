from config import tau0, tau1


def selection_rule(X_j_val: float, cum_selected: int) -> bool:
    return X_j_val >= (tau1 + cum_selected / tau0)

def selection_rule_2(X_j_val: float, cum_selected: int, j: int, t: int) -> bool:
    if j < t:
        return X_j_val < (cum_selected / tau0)
    else:
        return tau1 < cum_selected 

def selection_rule_final(mu_t: float, cum_selected: int) -> bool:
    return mu_t < (tau1 + cum_selected / tau0)