#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 14:53:40 2017
# Machine learning logistics regression functions:
    logreg_param, logreg_hyp, logreg_cost, logreg_cost_grad,
    logreg_prob.
@author: bernardoalencar
"""
import numpy as np
from ML_Regression import grad_descent

def logreg_param(dataX: 'pd.DataFrame or similar',
                 dataY: 'pd.DataFrame or similar',
                 alpha: float = 10**(-2)) -> list:
    try:
        X = np.matrix(dataX)
        Y = np.matrix(dataY)
    except:
        return print('dataSet cannot be converted into Matrix')
    n = X.shape[0]
    #Adding a column of ones:
    X = np.concatenate((np.ones((n,1)),X), axis = 1)
    #Gradient descent in case n is too large.
    w = [0] * Y.shape[1]
    for i in range(Y.shape[1]):
        w[i] = grad_descent(X,Y[:,i], logreg_cost,
                           logreg_cost_grad, alpha = alpha)[0]        
    return w

def logreg_hyp(X: np.matrix, w: np.matrix) -> np.matrix:
    if X[:,1].any() != 1:
        X = np.concatenate((np.ones((X.shape[0],1)),X), axis = 1)
    hyp = 1/(1 + np.exp(-(X*w)))
    return hyp

def logreg_cost(X: np.matrix, Y: np.matrix,
                w: np.matrix, reg_factor: float = 0) -> float:
    n = X.shape[0]
    hyp = logreg_hyp(X,w)
    cost = 1/(n) * (-Y.T * np.log(hyp) -(1-Y).T*np.log(1-hyp) +
              reg_factor * (w[1:].T * w[1:]))
    return cost

def logreg_cost_grad(X: np.matrix, Y: np.matrix,
                     w: np.matrix, reg_factor: float = 0) -> np.matrix:
    n = X.shape[0]
    hyp = logreg_hyp(X,w)
    cost_grad = (1/n) * (X.T * (hyp - Y) + reg_factor * w)
    return cost_grad


def logreg_prob(X: np.matrix, w: np.matrix) -> np.matrix:
    X = np.matrix(X)
    n = X.shape[0]
    #Adding a column of ones:
    X = np.concatenate((np.ones((n,1)),X), axis = 1)
    pred = logreg_hyp(X,w[0])
    for i in range(1,len(w)):
        pred = np.concatenate((pred,logreg_hyp(X,w[i])), axis = 1)
    return pred