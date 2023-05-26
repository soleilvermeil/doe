import numpy as np
import sympy
import scipy
from typing import Union
import matplotlib.pyplot as plt
import pandas as pd

class LinearModel:
    def __init__(self, modelspec: np.ndarray, X: np.ndarray, y: np.ndarray, coefficients: np.ndarray, residuals: np.ndarray):
        if isinstance(modelspec, str):
            modelspec = model_matrix(name=modelspec, factors=X.shape[1])
        self.modelspec = modelspec
        self.X = X
        self.y = y
        self.coefficients = coefficients
        self.residuals = residuals

    def predict(self, X: Union[list, np.ndarray]) -> np.ndarray:
        """
        Returns the predicted response of a multiple linear regression model with given coefficients, for the data matrix X.

        Parameters
        ----------
        X : np.ndarray
            The matrix of predictors.

        Returns
        -------
        np.ndarray
            The predicted response.
        """
        if isinstance(X, list):
            X = np.array(X)
        if len(np.shape(X)) == 1:
            X = X.reshape(-1, 1)
        XX = x2fx(X=X, modelspec=self.modelspec)
        return XX @ self.coefficients
    
    def relative_effect(self) -> np.ndarray:
        """
        Returns the relative effect of each predictor variable on the response variable.

        Returns
        -------
        np.ndarray
            The relative effect of each factor, normalized by the effect of the constant term.
        """
        XX = x2fx(X=self.X, modelspec=self.modelspec)
        delta = XX.max(axis=0) - XX.min(axis=0)
        a0 = self.coefficients[0]
        return self.coefficients[1:] * delta[1:] / a0
    
    def rmse(self) -> float:
        """
        Returns the root mean squared error of the model.

        Returns
        -------
        float
            The root mean squared error of the model.
        """
        return np.sqrt(np.mean((self.y - self.predict(X=self.X)) ** 2))
    
    def coef_names(self) -> list:
        """
        Returns the names of the coefficients of a model.

        Returns
        -------
        list
            The names of the coefficients.
        """
        names = []
        for row in self.modelspec:
            name = ""
            for i, power in enumerate(row):
                if power == 0:
                    continue
                elif power == 1:
                    name += f":x{i+1}"
                else:
                    name += f":x{i+1}^{power}"
            if name == "":
                name = ":x0"
            names.append(name[1:])
        return names
        

def model_matrix(name: str, factors: int) -> np.ndarray:
    """
    Returns a model matrix for a given model name and number of factors.

    Parameters
    ----------
    name : str
        The name of the model. Can be one of "linear", "interaction", "quadratic", "purequadratic".
    factors : int
        The number of factors.

    Returns
    -------
    np.ndarray
        The model matrix.
    """
    constant = np.zeros(shape=(1, factors))
    linear = np.eye(N=factors)
    interaction = np.array([x for x in sympy.utilities.iterables.multiset_permutations([0] * (factors-2) + [1] * 2)])[::-1]
    quadratic = 2 * np.eye(N=factors)

    if name == "linear":
        return np.append(constant, linear, axis=0).astype(int)
    elif name == "interaction":
        return np.append(np.append(constant, linear, axis=0), interaction, axis=0).astype(int)
    elif name == "quadratic":
        return np.append(np.append(np.append(constant, linear, axis=0), interaction, axis=0), quadratic, axis=0).astype(int)
    elif name == "purequadratic":
        return np.append(np.append(constant, linear, axis=0), quadratic, axis=0).astype(int)
    else:
        raise ValueError("Invalid model name.")
    
def x2fx(X: Union[list, np.ndarray], modelspec: Union[str, list, np.ndarray]) -> np.ndarray:
    """
    Converts a matrix of predictors X to a design matrix D for regression analysis. Distinct predictor variables should appear in different columns of X.

    Parameters
    ----------
    X : Union[list, np.ndarray]
        The matrix of predictors.
    model : Union[str, list, np.ndarray]
        The model matrix.

    Returns
    -------
    np.ndarray
        The design matrix.
    """

    if isinstance(X, list):
        X = np.array(X)
    if isinstance(modelspec, str):
        modelspec = model_matrix(name=modelspec, factors=X.shape[1])
    elif isinstance(modelspec, list):
        modelspec = np.array(modelspec)
    elif isinstance(modelspec, np.ndarray):
        assert X.shape[1] == modelspec.shape[1], "The number of columns in 'X' and 'model' must be equal."

    M = np.ones(shape=(X.shape[0], modelspec.shape[0]))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            for alpha in range(X.shape[1]):
                M[i, j] *= X[i, alpha] ** modelspec[j, alpha]

    return M

def fitlm(X: Union[list, np.ndarray], y: Union[list, np.ndarray], modelspec: Union[str, list, np.ndarray] = "linear") -> LinearModel:
    """
    Returns the linear model fit to the data matrix X.

    Parameters
    ----------
    X : Union[list, np.ndarray]
        The matrix of inputs.
    y : Union[list, np.ndarray]
        The response vector.
    modelspec : Union[str, list, np.ndarray]
        The model matrix.

    Returns
    -------
    LinearModel
        The fitted linear model.
    """
    if isinstance(X, list):
        X = np.array(X)
    if isinstance(y, list):
        y = np.array(y).reshape(-1, 1)
    XX = x2fx(X, modelspec)
    coefficients, residuals, _, _ = np.linalg.lstsq(XX, y, rcond=None)
    coefficients = coefficients.reshape(-1)
    return LinearModel(modelspec=modelspec, X=X, y=y, coefficients=coefficients, residuals=residuals)