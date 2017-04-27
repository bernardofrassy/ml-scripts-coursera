#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 13:25:55 2017

@author: bernardoalencar
"""
from NeuralNetwork import neural_net_param, nn_prediction
from DataPreProcess import data_seg
from DataSource import X_seg, Y_multi_seg, Y0_seg, X0_seg, X1_seg, Y1_seg, X,Y
from ML_LogisticsRegression import logreg_param,logreg_prob
import pandas as pd
import numpy as np


def all_comb(n: int, k: int) -> list:
    """
    Create all possible combinations for 'n' items with 'k'
    possible classes.
    """
    numComb = k ** n
    allComb = list()
    currComb = list()
    for value in range(0,numComb):
        currComb = [(value // k ** i) % (k) + 1 for i in range(0,n)]
        allComb.append(currComb)        
    return allComb

def test_nn_arch(X_seg: list,Y_seg: list, maxNeurons: int, maxLayers: int,
            alpha: float = 2, reg_factor: float = 0.5, **kargs) -> list:
    """
    Tests all possibles architectures for a neural network. Prints the result
    in a file named 'test_nn_result'.
    """
    archs = []
    yields = []
    arch_yields = {}
    for numLayers in range(1,maxLayers+1):
        allArchs = all_comb(numLayers,maxNeurons)
        for arch in allArchs:
            try:
                w = neural_net_param(X_seg[0],Y_seg[0], numLayers, arch, alpha,
                                     reg_factor, **kargs)[0]
                prob_Y = nn_prediction(X_seg[1],w)
                pred_Y = (prob_Y > 0.5) * (1)
                rightRatio = np.mean((pred_Y == Y_multi_seg[1]) * (1),
                                     axis = (0,1))
                archs.append(arch)
                yields.append(rightRatio)
                print(arch, ' is OK')
            except:
                print(arch,' is not OK')
        arch_yields = [(archs[i],yields[i]) for i in range(len(archs))]
    arch_yields = sorted(arch_yields, key = lambda x:-x[1])
    with open('test_nn_result','w') as f:
        f.write('{0:>9}{1:>9}\n'.format('Arch','Yield'))
        for i in range(len(arch_yields)):
            f.write('{0:>9}{1:>9.6f}\n'.format(repr(arch_yields[i][0]),
                                              float(arch_yields[i][1])))
    return arch_yields

def test_param(X_cross: np.array, Y_cross: np.array, w_param: np.array or list,
               probFunction: 'function') -> (float,np.matrix):
    """
    Tests the parameters of logistic regression model or a neural network for 
    the percentage of success over a cross-evaluation set.
    """
    prob = probFunction(X_cross,w_param)
    if prob.shape[1] > 1:
        pred = ((prob == np.amax(prob, axis = 1)) * 1)
        numWrong = -np.sum((pred == Y_cross) * 1 - 1, axis = (0,1))/2
    else:
        pred = ((prob > 0.5) * 1)
        numWrong = -np.sum((pred == Y_cross) * 1 - 1, axis = (0,1))
    rightRatio = float(1 - numWrong/prob.shape[0])
    return rightRatio, pred

def plot_error_vs_examples(X: np.array, Y: np.array, paramFunction: 'function',
                           probFunction: 'function', **kargs) -> None:
    """
    **kargs will be passed to paramFunction.
    """
    n = X.shape[0]
    errorsPerExample = list()
#    try:
    for i in range(1,n//10-1):
        X_s, Y_s = data_seg(X[0:10*i,:], Y[0:10*i,:],
                                cut_values = [0.6,0.2,0.2])
        w = paramFunction(X_s[0],Y_s[0], **kargs)
        rightRatio = test_param(X_s[1],Y_s[1], w, probFunction)[0]
        errorsPerExample.append((int(10*i),rightRatio))
#    except:
#        return print('Not able to slice given data')
    return errorsPerExample
            
            


#w = logreg_param(X_seg[0],Y_multi_seg[0])

#result, pred = test_param(X_seg[1],Y_multi_seg[1], w,
#                          probFunction = logreg_prob)

#w_nn = neural_net_param(X_seg[0], Y_multi_seg[0], 3, [3,3,3])[0]

#result_nn, pred_nn = test_param(X_seg[1],Y_multi_seg[1], w_nn, nn_prediction)

#plot_logreg = plot_error_vs_examples(X,Y, logreg_param, logreg_prob)

plot_nn = plot_error_vs_examples(X,Y, neural_net_param, nn_prediction,
                                 numLayers = 3, numNeurons = [3,3,3])

