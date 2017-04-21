#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 15:08:32 2017

@author: bernardoalencar
"""
from NeuralNetwork import (forward_propagation, check_shape, neural_net_cost,
                           grad_descent_nn, neural_net_param)
import numpy as np
import sklearn
import pdb

iris = sklearn.datasets.load_iris()
X = iris.data[:, :2]
Y = iris.target
Y_1 = np.matrix(Y.copy() * (Y == 1)).T
Y_2 = np.matrix((Y.copy())/2 * (Y == 2)).T
Y_0 = np.matrix((Y.copy()+1) * (Y == 0)).T
Y = np.matrix(Y.copy()).T

def backpropagation(X,Y,theta_list,neuron_list, reg_factor = 0):
    numLayers = len(theta_list)-1
    # Defining partial derivative to z:
    delta = [0]*(len(theta_list))
    dg_z = [0] * len(theta_list)
    for i in range(0,len(theta_list)):
        dg_z[i] = np.multiply(neuron_list[i+1],(1-neuron_list[i+1]))
    # Calculating deltas:
    for i in range(0,numLayers+1)[::-1]:
        if i == (numLayers):
            delta[i] = np.multiply(neuron_list[i+1] - Y,dg_z[i])
        else:
            delta[i] = np.multiply((delta[i+1]* theta_list[i+1].T),dg_z[i])
    gradient = [0] * len(theta_list)
    for i in range(0,len(theta_list)):
        gradient[i] = np.zeros(theta_list[i].shape)
    for i in range(1,numLayers+1):
        gradient[i] = (neuron_list[i].T * delta[i])/X.shape[0] 
        gradient[i] = gradient[i] + reg_factor * theta_list[i]
    return gradient


hyp, theta_list, neuron_list, w = neural_net_param(X,Y_2,3,[4,5,6])
forward_propagation(X,w)[1]
grad = backpropagation(X,Y,theta_list,neuron_list)[0]
check_shape(grad)