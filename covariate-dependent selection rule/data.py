import numpy as np
from config import beta


def generate_data(n: int, seed: int):
    """
    Generate online data of size n:
    - X ~ Uniform(0,2)
    - Y = X * beta + noise, where noise ~ N(0, sqrt(X/2))
    Returns:
        X: np.ndarray, shape (n,1)
        Y: np.ndarray, shape (n,)
        mu: np.ndarray, shape (n,)
    """
    """
    这里我们引入rng是为了让实验结果可复现, 如果只用一个np.random不用rng的话, 那么每次实验结果都会不同。
    """
    rng_x = np.random.default_rng(seed)
    rng_y=np.random.default_rng(seed+1)
    X = rng_x.uniform(0, 2, size=(n, 1))
    mu = X[:, 0] * beta
    sigma = np.sqrt(X[:, 0] / 2)
    Y = mu + rng_y.normal(0, sigma, size=n)
    return X, Y, mu