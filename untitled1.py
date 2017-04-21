#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:32:59 2017

@author: bernardoalencar
"""
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets.samples_generator import make_regression 
import numpy as np
import pdb
import copy

iris = sklearn.datasets.load_iris()
X = iris.data[:, :2]
Y = iris.target
Y_1 = np.matrix(Y.copy() * (Y == 1)).T
Y_2 = np.matrix((Y.copy())/2 * (Y == 2)).T
Y_0 = np.matrix((Y.copy()+1) * (Y == 0)).T
Y = np.matrix(Y.copy()).T

def forward_propagation(X,theta_list):
    neuron_list = [0]*(len(theta_list)+1)
    z_list = [0]*(len(theta_list)+1)
#    pdb.set_trace()
    for i in range(0,len(theta_list)+1):
        if i == 0:
            neuron_list[i] = X
        else:
            z_list[i-1] = neuron_list[i-1] * theta_list[i-1]
            neuron_list[i] = 1/(1 + np.exp(-z_list[i-1]))
    hyp = neuron_list[-1]
    return neuron_list, hyp
    
def check_shape(array):
    for i in range(0,len(array)):
        print('For %s shape is: %s' %(i,array[i].shape))
    return

def backpropagation(X,Y,theta_list,neuron_list, reg_factor = 0):
    numLayers = len(theta_list)-1
    # Defining partial derivative to z:
    delta = [0]* len(theta_list)
    dg_z = [0] * len(theta_list)
    for i in range(0,len(theta_list)):
        dg_z[i] = np.multiply(neuron_list[i+1],(1-neuron_list[i+1]))
    # Calculating deltas:
    for i in range(0,numLayers+1)[::-1]:
        if i == (numLayers):
            delta[i] = neuron_list[i+1] - Y
        else:
            delta[i] = np.multiply(delta[i+1],dg_z[i+1]) * theta_list[i+1].T
    gradient = [0] * len(theta_list)
    for i in range(0,len(theta_list)):
        gradient[i] = np.zeros(theta_list[i].shape)
    for layer in range(0,len(gradient)):
        for n in range(0,X.shape[0]):
            gradient[layer] = neuron_list[layer][i].T * delta[layer][i]
    return gradient

def neural_net_cost(X,Y, theta_list, reg_factor = 0):
    hyp = forward_propagation(X,theta_list)[1]
    m = X.shape[0]
    theta_sum = [np.sum(np.sum(np.square(theta_list[i]))) for i in range(0,len(theta_list))]
    theta_sum = np.sum(theta_sum)
    cost = -(Y.T * np.log(hyp) + (1-Y.T) * np.log(1 - hyp))/m
    cost = cost + reg_factor/(2*m) * theta_sum
    cost = np.sum(cost)
    print('Total:', cost,'1:', Y.T * np.log(hyp), '2:', (1-Y.T) * np.log(1 - hyp))
#    pdb.set_trace()
    return cost

def grad_descent_nn(X, Y, costFunction, gradFunction, neuron_list,
                    w_initial = 0, alpha = 10**(-2), reg_factor = 0,maxIteration = 100):
    import numpy as np 
    # Initial guess (all zeros):
    if w_initial == 0:
        w_initial = np.matrix(np.zeros((X.shape[1],1)))
    w = w_initial    
    # Apply gradient descent:
    count = 0
    error = 0.1
    cost_old = 10
    while ((error > 10**(-10)) and (count < maxIteration)):
        print(count)
        cost = costFunction(X,Y, w, reg_factor = reg_factor)
        grad = gradFunction(X,Y,w, neuron_list = neuron_list, reg_factor = reg_factor)
        w_new = np.subtract(w, np.multiply(grad, alpha))
        #In case the cost Function increases:
        if cost > cost_old:
            print('Cost function is increasing. Code will stop.')
        error = abs(cost - cost_old)
        w = w_new
        cost_old = cost
        count += 1  
        print(cost)
        print((error > 10**(-10)), (count < maxIteration))
    return w,cost

def neural_net_param(dataX, dataY, numLayers, numNeurons, reg_factor = 0):
    try:
        X = np.matrix(dataX)
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
    # Defining parameters initial values:
    theta_list = [0]*(numLayers + 1)
    
    # Random inital theta_values:
    for i in range(0,len(theta_list)):
        if i == 0:
            theta_list[i] = 2*(np.random.rand(X.shape[1], numNeurons[i])-0.5)
        else:
            theta_list[i] = 2*(np.random.rand(numNeurons[i-1], numNeurons[i])-0.5)
    neuron_list, hyp = forward_propagation(X,theta_list)
    w = grad_descent_nn(X, Y, neural_net_cost, backpropagation, neuron_list,
                         w_initial = theta_list)[0]
    return hyp,theta_list,neuron_list, w

hyp, theta_list, neuron_list, w = neural_net_param(X,Y_2,3,[4,5,6])
forward_propagation(X,w)


#def gradient_check(costFunction, X,Y, theta_list, reg_factor = 0):
#    eps = 10**(-10)
#    upper_theta = copy.deepcopy(theta_list)
#    lower_theta = copy.deepcopy(theta_list)
#    grad_check = copy.deepcopy(theta_list)
#    for layer in range(0,len(theta_list)):
#        for n in range(0,theta_list[layer].shape[0]):
#            for d in range(0,theta_list[layer].shape[1]):
#                grad = 0
#                upper_theta[layer][n][d] += + eps
#                lower_theta[layer][n][d] -=  eps
#                grad= (costFunction(X,Y,upper_theta) - costFunction(X,Y,lower_theta))/ (2*eps)
#                grad_check[layer][n][d] = grad
#                upper_theta[layer][n][d] -= eps
#                lower_theta[layer][n][d] += eps
#    return grad_check

X = np.matrix(X)

                
