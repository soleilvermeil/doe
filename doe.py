import numpy as np
import sympy as sp
from typing import Union

__author__ = "Julien L."
__licence__ = "MIT"
__email__ = "julien.l___@epfl.ch"

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
    constant = np.ones(shape=(1, factors))
    linear = np.eye(N=factors)
    interaction = np.array([x for x in sp.utilities.iterables.multiset_permutations([0] * (factors-2) + [1] * 2)])
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
    
def x2fx(X: np.ndarray, model: Union[str, np.ndarray]) -> np.ndarray:
    """
    Converts a matrix of predictors X to a design matrix D for regression analysis. Distinct predictor variables should appear in different columns of X.

    Parameters
    ----------
    X : np.ndarray
        The matrix of predictors.
    model : Union[str, np.ndarray]
        The model matrix.

    Returns
    -------
    np.ndarray
        The design matrix.
    """
    if isinstance(model, str):
        model = model_matrix(name=model, factors=X.shape[1])
    elif isinstance(model, np.ndarray):
        assert X.shape[1] == model.shape[1], "The number of columns in 'X' and 'model' must be equal."

    M = np.ones(shape=(X.shape[0], model.shape[0]))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            for alpha in range(X.shape[1]):
                M[i, j] *= X[i, alpha] ** model[j, alpha]

    return M

def fitlm(X: np.ndarray, y: np.ndarray, model: Union[str, np.ndarray]) -> np.ndarray:
    """
    Returns the coefficients of a multiple linear regression model of the response y, fit to the data matrix X.

    Parameters
    ----------
    X : np.ndarray
        The matrix of predictors.
    y : np.ndarray
        The response vector.
    model : Union[str, np.ndarray]
        The model matrix.

    Returns
    -------
    np.ndarray
        The coefficients of the fitted model.
    """
    XX = x2fx(X, model)
    a, residuals, _, _ = np.linalg.lstsq(XX, y, rcond=None)
    return a.reshape(-1)

def predict(X: np.ndarray, coefficients: np.ndarray, model: Union[str, np.ndarray]) -> np.ndarray:
    """
    Returns the predicted response of a multiple linear regression model with given coefficients, for the data matrix X.

    Parameters
    ----------
    X : np.ndarray
        The matrix of predictors.
    a : np.ndarray
        The coefficients of the fitted model.
    model : Union[str, np.ndarray]
        The model matrix.

    Returns
    -------
    np.ndarray
        The predicted response.
    """
    if len(np.shape(X)) == 1:
        X = X.reshape(1, -1)
    XX = x2fx(X, model)
    return XX @ coefficients

def anova(X: np.ndarray, y: np.ndarray, model: Union[np.ndarray, str]) -> dict:
    return None