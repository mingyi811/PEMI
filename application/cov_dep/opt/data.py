import numpy as np
from config import beta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
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
    data_df = pd.read_csv('data/davis_other_data_models.csv')
    
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
    muhat_1 = np.array(selected_df['muhat_1'])
    muhat_2 = np.array(selected_df['muhat_2'])
    muhat_3 = np.array(selected_df['muhat_3'])
    
    return Y, muhat_1, muhat_2, muhat_3



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
#     """
#     """
#     if n == 0:
#         return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

#     rng_x = np.random.default_rng(seed)
#     rng_y=np.random.default_rng(seed+1)
#     X = rng_x.uniform(0, 2, size=(n, 1))
#     mu = X[:, 0] * beta
#     sigma = np.sqrt(X[:, 0] / 2)
#     Y = mu**3 + 10*mu**2 + mu + rng_y.normal(0, sigma, size=n)

#     rng_x_train=np.random.default_rng(seed+2)
#     rng_y_train=np.random.default_rng(seed+3)
#     X_train = rng_x_train.uniform(0, 2, size=(n, 1))
#     mu_train = X_train[:, 0] * beta
#     sigma_train = np.sqrt(X_train[:, 0] / 2)
#     Y_train = mu_train + rng_y_train.normal(0, sigma_train, size=n)  # 1D
#     mu_test_linear = predict(X_train, Y_train, X, model_type="linear")
#     mu_test_random_forest = predict(X_train, Y_train, X, model_type="random_forest")
#     mu_test_polynomial = predict(X_train, Y_train, X, model_type="polynomial")
#     return X, Y, mu_test_linear, mu_test_random_forest, mu_test_polynomial


# def predict(x_train, y_train, x_test, model_type: str = "linear"):
#     """
#     Fits a linear regression model or a random forest model or a polynomial regression model using training data and predicts on test data.
    
#     Parameters:
#     x_train: array-like, training features
#     y_train: array-like, training target
#     x_test: array-like, test features
    
#     Returns:
#     mu_test: array-like, predicted values on test data
#     """

#     if model_type == "linear":
#         model = LinearRegression()
#         model.fit(x_train, y_train)
#         mu_test = model.predict(x_test)
#     elif model_type == "random_forest":
#         model = RandomForestRegressor()
#         model.fit(x_train, y_train)
#         mu_test = model.predict(x_test)
#     elif model_type == "polynomial":
#         poly = PolynomialFeatures(degree=2)
#         x_train_poly = poly.fit_transform(x_train)
#         x_test_poly = poly.transform(x_test)
#         model = LinearRegression()
#         model.fit(x_train_poly, y_train)
#         mu_test = model.predict(x_test_poly)
#     else:
#         raise ValueError(f"Invalid model type: {model_type}")
#     return mu_test