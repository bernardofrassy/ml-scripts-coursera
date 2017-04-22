#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 12:51:14 2017

@author: bernardoalencar
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 14:32:49 2017

@author: bernardoalencar

Machine Learning Functions
"""
import numpy as np
import pdb
import sklearn
import pandas as pd
from sklearn.datasets.samples_generator import make_regression 
from ML_Regression import data_seg

iris = sklearn.datasets.load_iris()
X = iris.data[:, :2]
X = np.matrix(X)
Y = iris.target
Y_1 = np.matrix(Y * (Y == 1)).T
Y_2 = np.matrix(Y * (Y == 2)).T
Y_0 = np.matrix((Y + 1) * (Y == 0)).T
Y = np.matrix(Y).T

data = pd.DataFrame(X)
data['Y'] = Y
data_train, data_cross, data_test = data_seg(data)
Y_train = np.matrix(data_train['Y']).T
Y_cross = np.matrix(data_cross['Y']).T
Y_test = np.matrix(data_test['Y']).T

Y0_train = (Y_train == 0) * 1
Y0_cross = (Y_cross == 0) * 1
Y0_test = (Y_test == 0) * 1

Y1_train = (Y_train == 1) * 1
Y1_cross = (Y_cross == 1) * 1
Y1_test = (Y_test == 1) * 1

Y2_train = (Y_train == 2) * 1
Y2_cross = (Y_cross == 2) * 1
Y2_test = (Y_test == 2) * 1

X_train =  np.matrix(data_train.iloc[:,:-1])
X_cross =  np.matrix(data_cross.iloc[:,:-1])
X_test =  np.matrix(data_test.iloc[:,:-1])

def neural_net_hyp(dataX, dataY, numLayers, numNeurons, reg_factor = 0):
    try:
        X = np.matrix(dataX)
        X = np.concatenate((np.ones((X.shape[0],1)),X), axis = 1)
    except:
        raise ValueError('dataSet cannot be converted into Matrix')
    try:
        numLayers = int(numLayers)
    except:
        print('numLayers must be an integer or float that can be coerced into an integer')
        return
    try:
        numNeurons = [int(numNeurons)] * numLayers
        numNeurons.append(int(1))
    except:
        try:
            numNeurons = list(int(i) for i in numNeurons)
            numNeurons.append(int(1))
        except:
            print('NumNeurons must be a single value or a vector of size numLayers')
            return
    if len(numNeurons)-1 != numLayers:
        raise ValueError('NumNeurons must be a single value or a vector of size numLayers')
    #Defining parameters initial values:
    theta_list = [0]*(numLayers+1)
    neuron_list = [0]*(numLayers + 1)
    z_list = [0]*(numLayers + 1)
    #Defining theta values:
    for i in range(len(theta_list)):
        if i == 0:
            theta_list[i] = np.random.rand(X.shape[1], numNeurons[i])
        else:
            theta_list[i] = np.random.rand(numNeurons[i-1]+1, numNeurons[i])
    #Defining neuron and z values:
    for i in range(numLayers+1):
        if i == 0:
            z_list[i] = X * theta_list[i]
        else:
            z_list[i] = neuron_list[i-1] * theta_list[i]
        neuron_list[i] = 1/(1 + np.exp(-z_list[i]))
        neuron_list[i] = np.concatenate((np.ones((X.shape[0],1)),X), axis = 1)
    return theta_list,neuron_list

def forward_propagation(X,theta_list):
    neuron_list = [0]*(len(theta_list)+1)
    z_list = [0]*(len(theta_list)+1)
    for i in range(len(theta_list)+1):
        if i == 0:
            neuron_list[i] = X
        else:
            z_list[i-1] = neuron_list[i-1] * theta_list[i-1]
            neuron_list[i] = 1/(1 + np.exp(-z_list[i-1]))
            if i < len(theta_list):
                ones_row = np.ones((neuron_list[i].shape[0],1))
                neuron_list[i] = np.concatenate((ones_row,neuron_list[i]), axis = 1)
    return neuron_list
    
def check_shape(array):
    for i in range(len(array)):
        print('For %s shape is: %s' %(i,array[i].shape))
    return

def backpropagation(X,Y,theta_list,neuron_list, reg_factor = 0):
    numLayers = len(theta_list)-1
    # Defining partial derivative to z:
    delta = [0]*(len(theta_list))
    dg_z = [0] * len(theta_list)
    for i in range(len(theta_list)):
        dg_z[i] = np.multiply(neuron_list[i+1],(1-neuron_list[i+1]))
    # Calculating deltas:
    for i in range(numLayers+1)[::-1]:
        if i == (numLayers):
            delta[i] = neuron_list[i+1] - Y
        else:
            delta[i] = np.multiply((delta[i+1]* theta_list[i+1][1:,:].T),dg_z[i][:,1:])
    gradient = [0] * len(theta_list)
    for i in range(len(theta_list)):
        gradient[i] = np.zeros(theta_list[i].shape)
    for i in range(numLayers+1):
        gradient[i] = (neuron_list[i].T * delta[i])/X.shape[0] 
        gradient[i] = gradient[i] + reg_factor * theta_list[i]
    return gradient

def neural_net_cost(X,Y, theta_list, neuron_list, reg_factor = 0):
    hyp = neuron_list[-1]
    m = X.shape[0]
    theta_sum = [np.sum(np.sum(np.square(theta_list[i]))) for i in range(len(theta_list))]
    theta_sum = np.sum(theta_sum)
    cost = (-Y.T * np.log(hyp) - (1-Y).T * np.log(1 - hyp))/m
    cost = cost + reg_factor/m * theta_sum
    cost = np.sum(cost)
    return cost

def grad_descent_nn(X, Y, costFunction, gradFunction,
                    w_initial = 0, alpha = 10**(-2), reg_factor = 0,maxIteration = 100000):
    import numpy as np 
    #Initial guess (all zeros):
    if w_initial == 0:
        w_initial = np.matrix(np.zeros((X.shape[1],1)))
    w = w_initial    
    #Apply gradient descent:
    count = 0
    count_increase = 0
    error = 0.1
    cost_old = 10
    while ((error > 10**(-10)) and (count < maxIteration)):
        neuron_list = forward_propagation(X,w)
        cost = costFunction(X,Y, w, neuron_list, reg_factor = reg_factor)
        grad = gradFunction(X,Y,w, neuron_list, reg_factor = reg_factor)
        w_new = np.subtract(w, np.multiply(grad, alpha))
        #In case the cost Function increases:
        if cost > cost_old:
            count_increase += 1
            if count_increase == (maxIteration * (10**(-3)) + 1):
                print('Cost function is increasing too frequently. Alpha reduced.')
                alpha = 0.5 * alpha
                count_increase = 0
        error = abs((cost - cost_old)/cost)
        w = w_new
        cost_old = cost
        count += 1 
    print(error, count)
    return w, grad, neuron_list

def neural_net_param(dataX, dataY, numLayers, numNeurons, alpha = 10**(-2), reg_factor = 0):
    try:
        X = np.matrix(dataX)
        X = np.concatenate((np.ones((X.shape[0],1)),X), axis = 1)
        Y = np.matrix(dataY)
    except:
        raise ValueError('dataSet cannot be converted into Matrix')
    try:
        numLayers = int(numLayers)
    except:
        print('numLayers must be an integer or float that can be coerced into an integer')
        return
    try:
        numNeurons = [int(numNeurons)] * numLayers
        numNeurons.append(int(1))
    except:
        try:
            numNeurons = list(int(i) for i in numNeurons)
            numNeurons.append(int(1))
        except:
            print('NumNeurons must be a single value or a vector of size numLayers')
            return
    if len(numNeurons)-1 != numLayers:
        raise ValueError('NumNeurons must be a single value or a vector of size numLayers')
    #Defining parameters initial values:
    theta_list = [0]*(numLayers + 1)
    
    #Random inital theta_values:
    for i in range(len(theta_list)):
        if i == 0:
            theta_list[i] = np.random.rand(X.shape[1], numNeurons[i])
        else:
            theta_list[i] = np.random.rand(numNeurons[i-1]+1, numNeurons[i])
    w,grad,neuron_list = grad_descent_nn(X, Y, neural_net_cost, backpropagation,
                                         w_initial = theta_list, alpha = alpha)
    return w, grad, neuron_list

def nn_prediction(X,theta_list):
    try:
        X = np.matrix(X)
        X = np.concatenate((np.ones((X.shape[0],1)),X), axis = 1)
    except:
        raise ValueError('dataSet cannot be converted into Matrix')
    neuron_list = [0]*(len(theta_list)+1)
    z_list = [0]*(len(theta_list)+1)
    for i in range(len(theta_list)+1):
        if i == 0:
            neuron_list[i] = X
        else:
            z_list[i-1] = neuron_list[i-1] * theta_list[i-1]
            neuron_list[i] = 1/(1 + np.exp(-z_list[i-1]))
            if i < len(theta_list):
                ones_row = np.ones((neuron_list[i].shape[0],1))
                neuron_list[i] = np.concatenate((ones_row,neuron_list[i]), axis = 1)
    return neuron_list


w, grad,neuron_list = neural_net_param(X_train,Y0_train,3,[3,3,3], alpha = 0.5, reg_factor = 0.5)
prob_y0 = nn_prediction(X_cross,w)[-1]
pred_y0 = (prob_y0 > 0.5) * (1)
print(sum((pred_y0 == Y0_cross) * (1))/Y0_cross.shape[0])

w, grad,neuron_list = neural_net_param(X_train,Y1_train,3,[3,3,3], alpha = 1, reg_factor = 0.5)
prob_y1 = nn_prediction(X_cross,w)[-1]
pred_y1 = (prob_y1 > 0.5) * (1)
print(sum((pred_y1 == Y1_cross) * (1))/Y1_cross.shape[0])

w, grad,neuron_list = neural_net_param(X_train,Y2_train,3,[3,3,3], alpha = 1, reg_factor = 0.5)
prob_y2 = nn_prediction(X_cross,w)[-1]
pred_y2 = (prob_y2 > 0.5) * (1)
print(sum((pred_y2 == Y2_cross) * (1))/Y2_cross.shape[0])