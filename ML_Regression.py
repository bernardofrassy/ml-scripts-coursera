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
from sklearn.datasets.samples_generator import make_regression 

def data_creation(examples, features):
    import numpy as np
    import pandas as pd
    n = examples
    d = features
    dataSet = pd.DataFrame()
    for i in range(1,d):
        if i < int(d):
            dataSet['x'+str(i)] = [(i*k)**i for k in range(n)]
        else:
            dataSet['xRand' + str(i+1)] = [np.random.normal() for i in range(n)]
    dataSet['y'] = [i*2 for i in range(n)]
    return dataSet

def data_seg(dataSet):
    import pandas as pd
    import random as rd
    dataLength = len(dataSet)
    try:
        df = pd.DataFrame(dataSet)
    except:
        return print('dataSet cannot be converted into pandas.DataFrame')
    df['Segment'] = pd.Series([rd.random() for i in range(dataLength)])
    dataTrain = df[(df['Segment'] <= 0.6)]
    dataTrain = dataTrain.iloc[:,:-1]
    dataCross = df[(df['Segment'] > 0.6) & (df['Segment'] <= 0.8)]
    dataCross = dataCross.iloc[:,:-1]
    dataTest = df[df['Segment'] > 0.8]
    dataTest = dataTest.iloc[:,:-1]
    return dataTrain,dataCross,dataTest

def normalize_data(data):
    for i in range(data.shape[1]):
        if np.std(data[:,i]) != 0:
            data[:,i] = (data[:,i] - np.average(data[:,i]))/np.std(data[:,i])
    return data

def lr_param(dataX,dataY, alpha = 10**(-2), maxIteration = 100000, reg_factor = None):
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

def ls_cost(X,Y,w, reg_factor = 0):
    n = X.shape[0]
    cost = 1/(2*n) * ((X * w - Y).T * (X * w - Y) + reg_factor * (w[1:].T * w[1:]))
    return cost

def ls_cost_grad(X,Y,w, reg_factor = 0):
    n = X.shape[0]
    cost_grad = (1/n) * (X.T * (X * w - Y) + reg_factor * w)
    return cost_grad

def grad_descent(X, Y, costFunction, gradFunction, alpha = 10**(-2), reg_factor = 0, maxIteration = 10000):
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

def plot_reg(X,Y,w):
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

X, Y = make_regression(n_samples=5000, n_features=1, n_informative=1, random_state=0, noise=35)
w = lr_param(X,Y, reg_factor = 1)[0]
plot_reg(X,Y,w)