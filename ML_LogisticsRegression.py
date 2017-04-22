#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 14:53:40 2017
# Machine learning logistics regression functions:
    logreg_param, logreg_hyp, logreg_cost, logreg_cost_grad,
    logreg_prob.
@author: bernardoalencar
"""

def logreg_param(dataX,dataY, alpha = 10**(-2)):
    import numpy as np
    try:
        X = np.matrix(dataX)
        Y = np.matrix(dataY).reshape([X.shape[0],1])
    except:
        return print('dataSet cannot be converted into Matrix')
    n = X.shape[0]
    #Adding a column of ones:
    X = np.concatenate((np.ones((n,1)),X), axis = 1)
    #Gradient descent in case n is too large.
    w = grad_descent(X,Y, logreg_cost,
                     logreg_cost_grad, alpha = alpha)[0]
    return w

def logreg_hyp(X,w):
    import numpy as np
    hyp = 1/(1 + np.exp(-(X*w)))
    return hyp

def logreg_cost(X,Y,w, reg_factor = 0):
    n = X.shape[0]
    hyp = logreg_hyp(X,w)
    cost = 1/(n) * (-Y.T * np.log(hyp) - (1-Y).T*np.log(1-hyp) + reg_factor * (w[1:].T * w[1:]))
    return cost

def logreg_cost_grad(X,Y,w, reg_factor = 0):
    n = X.shape[0]
    hyp = logreg_hyp(X,w)
    cost_grad = (1/n) * (X.T * (hyp - Y) + reg_factor * w)
    return cost_grad


def logreg_prob(X,w):
    import numpy as np
    X = np.matrix(X)
    n = X.shape[0]
    #Adding a column of ones:
    X = np.concatenate((np.ones((n,1)),X), axis = 1)
    prediction = logreg_hyp(X,w)
    return prediction

# Set binary Y to test function
Y_1 = Y * (Y == 1)
Y_2 = Y/2 * (Y == 2)
Y_0 = (Y+1) * (Y == 0)
w_0 = logreg_param(X,Y_0)
w_1 = logreg_param(X,Y_1)
w_2 = logreg_param(X,Y_2)
prob_0 = logreg_prob(X,w_0)
prob_1 = logreg_prob(X,w_1)
prob_2 = logreg_prob(X,w_2)

pred_0 = (np.greater(prob_0,prob_1) & np.greater(prob_0,prob_2))*1
pred_1 = (np.greater(prob_1,prob_0) & np.greater(prob_1,prob_2))*1
pred_2 = (np.greater(prob_2,prob_0) & np.greater(prob_2,prob_1))*1