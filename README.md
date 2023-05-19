# doe

MATLAB's design of experiment routines for Python

## How to use it

Put the file `doe.py` in the same folder than your project, and put `import doe` at the beginning of your code.

## Available functions

The following functions are already available. See [`example.ipynb`](https://github.com/SoleilVermeil/doe/blob/main/example.ipynb) to see how to use them.

* `doe.model_matrix(name, factors)` returns a model matrix for a given `model` name and number of `factors`.
* `doe.x2fx(X, model)` converts a matrix of predictors `X` to a design matrix for regression analysis.
* `doe.fitlm(X, y, model)` returns the coefficients of a multiple linear regression `model` of the response `y`, fit to the data matrix `X`.
* `doe.predict(X, coefficients, model)` returns the predicted response of a multiple linear regression `model` with given `coefficients`, for the data matrix `X`.

## Soon available

We are currently working on implementing the following functions. Do not hesitate to make pull requests if you want to contribute.

* `doe.anova(X, y, model)`
* `doe.normplot(coefficients)`
