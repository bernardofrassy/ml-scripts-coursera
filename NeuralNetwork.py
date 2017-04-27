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

def neural_net_hyp(dataX: 'pd.DataFrame or similar',
                   dataY: 'pd.DataFrame or similar',
                   numLayers: int, numNeurons: int,
                   reg_factor: float = 0) -> (list,list):
    """
    Calculates the weights and neurons value for a given neural network
    architecture.
    """
    try:
        X = np.matrix(dataX)
        X = np.concatenate((np.ones((X.shape[0],1)),X), axis = 1)
    except:
        raise ValueError('dataSet cannot be converted into Matrix')
    try:
        numLayers = int(numLayers)
    except:
        print('numLayers must be an integer or float that can' +
              ' be coerced into an integer')
        return
    try:
        numNeurons = [int(numNeurons)] * numLayers
        numNeurons.append(int(1))
    except:
        try:
            numNeurons = list(int(i) for i in numNeurons)
            numNeurons.append(int(1))
        except:
            print('NumNeurons must be a single value or a vector of size' +
                  ' numLayers')
            return
    if len(numNeurons)-1 != numLayers:
        raise ValueError('NumNeurons must be a single value or a vector' +
                         ' of size numLayers')
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

def forward_propagation(X: np.matrix, theta_list: list) -> list:
    """
    Performs forward propagation to calculate neuron values.
    """
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
    
def check_shape(a: list) -> None:
    """
    Prints shape of components inside the given argument.
    """
    for i in range(len(a)):
        print('For %s shape is: %s' %(i,a[i].shape))
    return

def backpropagation(X: np.matrix,Y: np.matrix,theta_list: list, 
                    neuron_list: list,
                    reg_factor: float = 0) -> list:
    """
    Calculates the gradient using backpropagation algorith for
    neural networks.
    """
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
            delta[i] = np.multiply((delta[i+1] * theta_list[i+1][1:,:].T), dg_z[i][:,1:])
    gradient = [0] * len(theta_list)
    for i in range(len(theta_list)):
        gradient[i] = np.zeros(theta_list[i].shape)
    for i in range(numLayers+1):
        gradient[i] = (neuron_list[i].T * delta[i])/X.shape[0] 
        gradient[i] = gradient[i] + reg_factor * theta_list[i]
    return gradient

def neural_net_cost(X: np.matrix,Y: np.matrix, theta_list: list,
                    neuron_list: list, reg_factor:float = 0) -> float:
    """
    Calculates the cross entropy for a given neural network.
    """
    cost = 0
    hyp = neuron_list[-1]
    m = X.shape[0]
    theta_sum = [np.sum(np.sum(np.square(theta_list[i]))) for i in
                 range(len(theta_list))]
    theta_sum = np.sum(theta_sum)
    cost = (- np.sum(np.multiply(Y, np.log(hyp)), axis = (0,1)) -
            np.sum(np.multiply((1-Y), np.log(1-hyp)), axis = (0,1)))/m
    cost = cost + reg_factor/m * theta_sum
    cost = np.sum(cost)
    return cost

def grad_descent_nn(X: np.matrix, Y: np.matrix,
                    costFunction: 'function', gradFunction: 'function',
                    w_initial: np.matrix, 
                    alpha: float = 10**(-2),reg_factor: float = 0,
                    maxIteration: int = 100000, printIteration: bool = False):
    """
    Performs gradient descent for a given neural network.
    """
    #Initial guess (all zeros):
    w = w_initial    
    #Apply gradient descent:
    count = 0
    count_increase = 0
    error = 0.1
    cost_old = 10
    w_new = [0] * len(w)
    while ((error > 10**(-10)) and (count < maxIteration)):
        neuron_list = forward_propagation(X,w)
        cost = costFunction(X,Y, w, neuron_list, reg_factor = reg_factor)
        grad = gradFunction(X,Y,w, neuron_list, reg_factor = reg_factor)
        if printIteration == True:
            print('Count ', count,'Cost: ', cost)
        for i in range(len(w)):
            w_new[i] = w[i] - grad[i] * alpha
        #In case the cost Function increases:
        if cost > cost_old:
            count_increase += 1
            if count_increase == (maxIteration * (10**(-4)) + 1):
                print('Cost function is increasing too frequently.' +
                      ' Alpha reduced.')
                alpha = 0.5 * alpha
                count_increase = 0
        error = abs((cost - cost_old)/cost)
        w = w_new
        cost_old = cost
        count += 1 
    if printIteration == True:
        print(error, count)
    return w, grad, neuron_list

def neural_net_param(dataX: 'pd.DataFrame or similar',
                     dataY: 'pd.DataFrame or similar',
                     numLayers: int, numNeurons: int,
                     alpha: float = 2, reg_factor: float = 0,
                     **kargs) -> (np.matrix, np.matrix, list):
    """
    Trains the weights for a given neural network architecture.
    """
    try:
        X = np.matrix(dataX)
        X = np.concatenate((np.ones((X.shape[0],1)),X), axis = 1)
        Y = np.matrix(dataY)
    except:
        raise ValueError('dataSet cannot be converted into Matrix')
    try:
        numLayers = int(numLayers)
    except:
        print('numLayers must be an integer or float that can be coerced' +
              ' into an integer')
        return
    try:
        numNeurons = [int(numNeurons)] * numLayers
        numNeurons.append(int(Y.shape[1]))
    except:
        try:
            numNeurons = list(int(i) for i in numNeurons)
            numNeurons.append(int(Y.shape[1]))
        except:
            print('NumNeurons must be a single value or a vector of size' +
                  ' numLayers')
            return
    if len(numNeurons)-1 != numLayers:
        raise ValueError('NumNeurons must be a single value or a vector' +
                         ' of size numLayers')
    #Defining parameters initial values:
    theta_list = [0]*(numLayers + 1)
    
    #Random inital theta_values:
    for i in range(len(theta_list)):
        if i == 0:
            theta_list[i] = np.random.rand(X.shape[1], numNeurons[i])
        else:
            theta_list[i] = np.random.rand(numNeurons[i-1]+1, numNeurons[i])
    w,grad,neuron_list = grad_descent_nn(X, Y, neural_net_cost,
                                         backpropagation,
                                         w_initial = theta_list, alpha = alpha,
                                         **kargs)
    return w

def nn_prediction(X: np.matrix,theta_list: list) -> list:
    try:
        X = np.matrix(X)
        X = np.concatenate((np.ones((X.shape[0],1)),X), axis = 1)
    except:
        raise ValueError('X cannot be converted into Matrix')
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
    return neuron_list[-1]