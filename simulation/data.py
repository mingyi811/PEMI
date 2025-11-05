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
from quantile_forest import RandomForestQuantileRegressor
from typing import Dict, Any
warnings.filterwarnings("ignore")

def generate_data(n: int, sigma: float, seed: int, model_type: str = "linear"):
    rng_x = np.random.default_rng(seed)
    rng_y=np.random.default_rng(seed+1)

    #setting 1: highly nonlinear data setting
    X = rng_x.uniform(low=-1, high=1, size=n + 1000).reshape(-1, 1)
    mu_x = (
        3.0 * np.sin(4.0 * np.pi * X[:, 0])
        + 4.0 * np.maximum(0.0, X[:, 0] - 0.3) ** 2
        - 4.0 * np.maximum(0.0, -(X[:, 0] + 0.4)) ** 2
    )

    Y = mu_x + rng_y.normal(size=n + 1000, scale=(0.5 + np.abs(X[:, 0])) * sigma)
    
    #setting 2: heteroscedastic data setting
    # X = rng_x.uniform(low=-1, high=1, size=(n+1000)*20).reshape((n+1000,20))
    # mu_x = (X[:,0] * X[:,1] + np.exp(X[:,3] - 1)) * 5
    # Y = mu_x + rng_y.normal(size=n+1000, scale=(np.abs(5.5 - abs(mu_x))/2 * sigma)) 
    
    X_train = X[:1000]
    Y_train = Y[:1000]
    X_test = X[1000:]
    Y_test = Y[1000:]
    if model_type == "linear":
        mu_test = predict(X_train, Y_train, X_test, model_type="linear")
    elif model_type == "random_forest":
        mu_test = predict(X_train, Y_train, X_test, model_type="random_forest")

    return X_test, Y_test, mu_test

def predict(x_train, y_train, x_test, model_type: str = "linear"):
    """
    Fits a linear regression model or a random forest model or a polynomial regression model using training data and predicts on test data.
    
    Parameters:
    x_train: array-like, training features
    y_train: array-like, training target
    x_test: array-like, test features
    
    Returns:
    mu_test: array-like, predicted values on test data
    """

    if model_type == "linear":
        model = LinearRegression()
        model.fit(x_train, y_train)
        mu_test = model.predict(x_test)
    elif model_type == "random_forest":
        model = RandomForestRegressor(max_depth=5, random_state=0)
        model.fit(x_train, y_train)
        mu_test = model.predict(x_test)
    elif model_type == "polynomial":
        poly = PolynomialFeatures(degree=2)
        x_train_poly = poly.fit_transform(x_train)
        x_test_poly = poly.transform(x_test)
        model = LinearRegression()
        model.fit(x_train_poly, y_train)
        mu_test = model.predict(x_test_poly)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    return mu_test





