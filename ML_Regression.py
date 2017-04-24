#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 14:50:07 2017
# Machine learning data treatment and linear regression functions: 
    data_creation,data_seg, normalize_data,
    lr_param, ls_cost, ls_cost_grad, grad_descent, plot_reg.
@author: bernardoalencar
"""
import numpy as np
import pandas as pd 

def lr_param(dataX: pd.DataFrame, dataY: pd.DataFrame,
             alpha: float = 10**(-2), maxIteration: int = 100000,
             reg_factor: float = None) -> np.matrix:
    """
    Calculates the parameters of a linear regression.
    Automatically includes the intercept.
    """
    import numpy as np
    try:
        X = np.matrix(dataX)
        Y = np.matrix(dataY).reshape([X.shape[0],1])
    except:
        return print('dataSet cannot be converted into Matrix')
    n = X.shape[0]
    #Adding a column of ones:
    X = np.concatenate((np.ones((n,1)),X), axis = 1)
    #Attempt to use analitcal response in small data sets:
    if n == 10**3:
        if reg_factor == None:
            w = ((X.T * X).I) * X.T * Y
            return w
        else:
            reg_matrix = reg_factor * np.matrix(np.identity(X.shape[1]))
            w = ((X.T * X + reg_matrix).I) * X.T * Y
            return w
    #Gradient descent in case n is too large.
    else:
        w = grad_descent(X,Y, ls_cost, ls_cost_grad, alpha = alpha, reg_factor = reg_factor)
    return w

def ls_cost(X: np.matrix,Y: np.matrix,w: np.matrix,
            reg_factor: float = 0) -> float:
    """
    Calculates the cost of using the least squares method.
    """
    n = X.shape[0]
    cost = 1/(2*n) * ((X * w - Y).T * (X * w - Y) + reg_factor * (w[1:].T * w[1:]))
    return cost

def ls_cost_grad(X: np.matrix,Y: np.matrix,w: np.matrix,
                 reg_factor: float = 0) -> np.matrix:
    """
    Calculates the cost gradient of a the least squares method.
    """
    n = X.shape[0]
    cost_grad = (1/n) * (X.T * (X * w - Y) + reg_factor * w)
    return cost_grad

def grad_descent(X: np.matrix, Y: np.matrix,
                 costFunction: 'function', gradFunction: 'fuction',
                 alpha: float = 10**(-2), reg_factor: float = 0,
                 maxIteration: int = 10000) -> (np.matrix,float):
    """
    Performs the gradient descent until a maxIteration is reached or if the
    cost value between iterations becomes smaller than 10**(-5).
    """
    import numpy as np 
    #Initial guess (all zeros):
    w = np.matrix(np.zeros((X.shape[1],1)))
    #Apply gradient descent:
    count = 0
    error = 0.1
    while ((error > 10**(-5)) and (count < maxIteration)):
        cost = costFunction(X,Y, w, reg_factor)
        grad = gradFunction(X,Y, w, reg_factor)
        w_new = w - alpha * grad
        #In case the cost Function increases:
        if costFunction(X,Y,w_new,reg_factor) > costFunction(X,Y,w,reg_factor):
            print('Cost function is increasing. Code will stop.')
        error = float(sum(abs(w_new - w))/w.shape[0])
        w = w_new
        count += 1   
    return w,cost

def plot_reg(X: np.matrix,Y: np.matrix,w: np.matrix) -> None:
    """
    Plots the results of a regression.
    """
    import matplotlib.pyplot as plt
    X = pd.DataFrame(X).copy()
    X['Intercept'] = 1
    cols_names = X.columns.tolist()
    cols_names = cols_names[-1:] + cols_names[:-1]
    X = X[cols_names]
    for col in range(1,X.shape[1]):
        plt.figure()
        plt.plot(X.iloc[:,col],list(Y),'x', label = 'Data')
        plt.legend()
        plt.title('Variable ' + str(cols_names[col])+ ' versus Y')
        y_predict = np.matrix(X.iloc[:,col]).T*w[col] + w[0]
        plt.plot(X.iloc[:,col],y_predict, label = 'Regression')
        plt.legend()
    return