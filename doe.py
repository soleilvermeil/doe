import numpy as np
import sympy
import scipy
from typing import Union
import matplotlib.pyplot as plt
import pandas as pd
import warnings

class LinearModel:
    def __init__(self, modelspec: np.ndarray, X: np.ndarray, y: np.ndarray, coefficients: np.ndarray, residuals: np.ndarray):
        if isinstance(modelspec, str):
            modelspec = model_matrix(name=modelspec, factors=X.shape[1])
        self.modelspec = modelspec
        self.X = X
        self.y = y
        self.coefficients = coefficients
        self.residuals = residuals

    def predict(self, x: Union[list, np.ndarray]) -> np.ndarray:
        """
        Returns the predicted response of a multiple linear regression model for the data matrix X.

        Parameters
        ----------
        X : np.ndarray
            The data matrix on which the model should try to predict the response.

        Returns
        -------
        np.ndarray
            The predicted response.
        """
        if isinstance(x, list):
            x = np.array(x)
        if len(np.shape(x)) == 1:
            x = x.reshape(-1, 1)
        XX = x2fx(X=x, modelspec=self.modelspec)
        return XX @ self.coefficients
    
    def relative_effect(self) -> np.ndarray:
        """
        Returns the relative effect of each coefficient.

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
        Returns the root mean squared error of the model's predicted response for the inputs that were used to fit the model.

        Returns
        -------
        float
            The root mean squared error.
        """
        return np.sqrt(np.mean((self.y - self.predict(x=self.X)) ** 2))
    
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
        

def modelspec2matrix(name: str, factors: int) -> np.ndarray:
    """
    Returns the matrix of a given model specification.

    Parameters
    ----------
    name : str
        The name of the model. Can be one of "linear", "interaction", "quadratic", "purequadratic".
    factors : int
        The number of factors.

    Returns
    -------
    np.ndarray
        The matrix of the model specification where each column corresponds to a factor, and each row corresponds to a term in the model.
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
    Converts a data matrix to a design matrix.

    Parameters
    ----------
    X : Union[list, np.ndarray]
        The data matrix.
    model : Union[str, list, np.ndarray]
        The model specification. It can either be a string (among "linear", "interaction", "quadratic", "purequadratic"), or an array where each column corresponds to a factor, and each row corresponds to a term in the model.

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

def fitlm(x: Union[list, np.ndarray], y: Union[list, np.ndarray], modelspec: Union[str, list, np.ndarray] = "linear") -> LinearModel:
    """
    Returns the linear model fit to the data matrix X.

    Parameters
    ----------
    X : Union[list, np.ndarray]
        The data matrix.
    y : Union[list, np.ndarray]
        The vector of responses.
    modelspec : Union[str, list, np.ndarray]
        The model specification. It can either be a string (among "linear", "interaction", "quadratic", "purequadratic"), or an array where each column corresponds to a factor, and each row corresponds to a term in the model.

    Returns
    -------
    LinearModel
        The fitted linear model.
    """
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y).reshape(-1, 1)
    X = x2fx(x, modelspec)
    coefficients, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
    coefficients = coefficients.reshape(-1)
    return LinearModel(modelspec=modelspec, X=x, y=y, coefficients=coefficients, residuals=residuals)

def normplot(data: Union[list, np.ndarray]) -> tuple[plt.Figure, plt.Axes]:
    """
    Returns a normal probability plot of the data.
    
    Parameters
    ----------
    data : Union[list, np.ndarray]
        The data to plot.
        
    Returns
    -------
    Union[plt.Figure, plt.Axes]
        The figure and axes of the plot.
    """
    def f(x, a, b):
        return np.exp(a + b * x) / (1 + np.exp(a + b * x))
    if isinstance(data, list):
        data = np.array(data)
    data = np.sort(data)
    X = data
    Y = np.linspace(0, 1, len(X)+2)[1:-1]
    fig, ax = plt.subplots()
    ax.plot(X, Y, "k+")
    ylim = ax.get_ylim()
    popt, _ = scipy.optimize.curve_fit(f, X, Y, sigma=X)
    X_fit = np.linspace(start=X.min(), stop=X.max(), num=10)
    Y_fit = f(X_fit, *popt)
    X_fit = X_fit[(Y_fit > 0) & (Y_fit < 1)]
    Y_fit = Y_fit[(Y_fit > 0) & (Y_fit < 1)]
    ax.plot(X_fit, Y_fit, "r-")
    ax.set_yscale("logit")
    ax.set_ylim(ylim)
    plt.xlabel("Data")
    plt.ylabel("Probability")
    return fig, ax

def ff2n(n: int) -> np.ndarray:
    """
    Returns a full factorial data matrix.

    Parameters
    ----------
    n : int
        The number of factors.

    Returns
    -------
    np.ndarray
        The data matrix.
    """
    if n == 1:
        return np.array([[-1], [1]])
    else:
        return np.append(
            np.append(-1 * np.ones(shape=(2 ** (n - 1), 1)), ff2n(n - 1), axis=1),
            np.append(np.ones(shape=(2 ** (n - 1), 1)), ff2n(n - 1), axis=1),
            axis=0
        )


def pbdesign(n: int) -> np.ndarray:
    """
    Returns a Plackett-Burman data matrix.

    Parameters
    ----------
    n : int
        The number of factors.

    Returns
    -------
    np.ndarray
        The data matrix.
    """
    D = scipy.linalg.hadamard(n)[:, 1:]
    return D[:n, :n - 1]

def fracfact(gen: str) -> np.ndarray:
    """
    Returns a fractional factorial data matrix.

    Parameters
    ----------
    gen : str
        The generator string, consisting of characters or group of characters for each factor.

    Returns
    -------
    np.ndarray
        The data matrix.
    """
    gen_splitted = gen.split(" ")
    gen_sorted = sorted(gen_splitted, key=len)
    number_of_primordial_columns = sum([len(g) == 1 for g in gen_sorted])
    F = ff2n(number_of_primordial_columns)
    G = np.zeros(shape=[np.shape(F)[0], len(gen_splitted)])
    for i in range(len(gen_splitted)):
        if len(gen_splitted[i]) == 1:
            G[:, i] = F[:, gen_sorted.index(gen_splitted[i])]
        else:
            G[:, i] = G[:, gen_splitted.index(gen_splitted[i][0])]
            for letter in gen_splitted[i][1:]:
                G[:, i] *= F[:, gen_splitted.index(letter)]
    return G

def t(confidence: float, dof: float) -> float:
    if confidence > 0.5:
        warnings.warn("Parameter 'confidence' corresponds here to the domain outside the interval, In the general case, this value is meant to be small. Be sure you are not confusing it.", category=Warning)
    return scipy.stats.t.ppf(1-confidence, dof)