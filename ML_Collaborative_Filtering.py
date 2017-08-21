"""Aplicando filtro colaborativo aos dados."""
import numpy as np
import pandas as pd


def collab_filtering(data: pd.DataFrame, n_features: int,
                     extra_features: pd.DataFrame = None,
                     alpha: float=10**(-2), momentum=True,
                     eps=0.9, maxIteration: int=100000,
                     lambdaFactor: float=0) -> (np.matrix, np.matrix):
    """Apply collaborative filtering, returns features and parameters."""
    n_prods = data.shape[0]
    n_users = data.shape[1]
    # Creating variables:
    Y = np.matrix(data)
    features = np.matrix(np.random.rand(n_prods, n_features))
    if extra_features is not None:
        n_extra_features = extra_features.shape[1]
        extraFeat = np.matrix(extra_features)
        X = np.concatenate((features, extraFeat), axis=1)
    else:
        n_extra_features = 0
        X = np.matrix(features)
    # Adding column of 1's:
    X = np.matrix(np.concatenate((np.ones((n_prods, 1)), X), axis=1))
    w = np.matrix(np.random.rand(n_features + n_extra_features + 1, n_users)
                  / 10)
    # Applying gradient descent:
    error_w = 1
    error_X = 1
    count = 0
    conv = True
    while conv:
        cost = cf_cost(X, Y, w)
        grad_w = cf_theta_grad(X, Y, w)
        grad_X = cf_X_grad(X, Y, w)
        if (not momentum) or (count == 0):
            wNew = w - alpha * grad_w
            xNew = X - alpha * grad_X
        if (momentum) and (count != 0):
            velo_w = wNew - w
            velo_X = xNew - X
            wNew = w + (eps * velo_w - (grad_w * alpha))
            xNew = X + (eps * velo_X - (grad_X * alpha))
        error_w = float(abs(wNew - w).sum()/w.shape[0])
        error_X = float(abs(xNew - X).sum()/X.shape[0])
        if cf_cost(xNew, Y, wNew, lambdaFactor) > cf_cost(X, Y, w,
                                                          lambdaFactor):
            print('Cost function is increasing. Code will stop.')
        print('Cost:', cost, 'Count:', count)
        w = wNew
        X = xNew
        conv = (error_w > 10**(-8)) | (error_X > 10**(-8)) | count < 10**5
        count += 1
    return w, X, cost


def cf_cost(X: np.matrix, Y: np.matrix, w: np.matrix,
            lambdaFactor: float=0) -> float:
    """Calculate the cost of using the least squares method."""
    n = X.shape[0]
    errors = (X * w - Y)
    errors[np.isnan(errors)] = 0
    print()
    cost = 1/(2*n) * (errors.T * errors + lambdaFactor * (w[1:].T * w[1:]))
    cost = np.sum(cost)
    return cost


def cf_theta_grad(X: np.matrix, Y: np.matrix, w: np.matrix,
                  lambdaFactor: float = 0) -> np.matrix:
    """Calculate the theta gradient for a least squares method."""
    n = X.shape[0]
    errors = (X * w - Y)
    errors[np.isnan(errors)] = 0
    cost_grad = (1/n) * (X.T * errors + lambdaFactor * w)
    return cost_grad


def cf_X_grad(X: np.matrix, Y: np.matrix, w: np.matrix,
              lambdaFactor: float = 0) -> np.matrix:
    """Calculate the X gradient for a least squares method."""
    n = X.shape[0]
    errors = (X * w - Y)
    errors[np.isnan(errors)] = 0
    cost_grad = (1/n) * (errors * w.T)
    return cost_grad
