import numpy as np
from config import beta
from DeepPurpose import utils, dataset, CompoundPred
import warnings
import pandas as pd
from DeepPurpose import DTI as models
warnings.filterwarnings("ignore")


def generate_data(n: int, seed: int):
    """
    Generate online data of size n from pre-generated DAVIS CSV:
    - Load the pre-generated CSV 'davis_other_data.csv'.
    - Randomly select n samples using the given seed for reproducibility.
    Returns:
        X_drugs: np.ndarray, shape (n,) - Drug SMILES strings
        X_targets: np.ndarray, shape (n,) - Target protein sequences
        Y: np.ndarray, shape (n,) - True affinity labels (pKd)
        muhat: np.ndarray, shape (n,) - Pre-computed model predictions
    """
    # Load the pre-generated CSV
    data_df = pd.read_csv('data/davis_other_data.csv')
    
    # Check if n exceeds available samples
    if n > len(data_df):
        raise ValueError(f"Requested n={n} exceeds available samples in CSV ({len(data_df)})")
    
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Randomly select n indices
    select_indices = np.random.permutation(len(data_df))[:n]
    
    # Select the rows
    selected_df = data_df.iloc[select_indices]
    
    Y = np.array(selected_df['Label'])
    muhat = np.array(selected_df['muhat'])
    
    return Y, muhat


# def generate_data(n: int, seed: int):
#     """
#     Generate online data of size n:
#     - X ~ Uniform(0,2)
#     - Y = X * beta + noise, where noise ~ N(0, sqrt(X/2))
#     Returns:
#         X: np.ndarray, shape (n,1)
#         Y: np.ndarray, shape (n,)
#         mu: np.ndarray, shape (n,)
#     """
#     rng_x = np.random.default_rng(seed)
#     rng_y=np.random.default_rng(seed+1)
#     X = rng_x.uniform(-0.5, 1.0, size=(n, 1))
#     mu = X[:, 0] * beta
#     #sigma = np.sqrt(X[:, 0] / 2)
#     sigma = np.abs(X[:, 0]) / 2
#     Y = mu + rng_y.normal(0, sigma, size=n)
#     return X, Y, mu