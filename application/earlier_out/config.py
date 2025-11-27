# Coverage level
alpha: float = 0.4
# Data generation slope
beta: float = 1.0
# Selection rule parameters

method: str = "weighted_quantile"
# Feature index to inspect
j_feature: int = 0
# Number of permutations for randomized testing
M: int = 1000
#quantile threshold
q: float = 0.7
#reference set method
reference_set_method: str = "ours"