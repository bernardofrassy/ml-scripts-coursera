#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 13:25:55 2017

@author: bernardoalencar
"""
from NeuralNetwork import neural_net_param, nn_prediction, check_shape
from DataSource import X_seg, Y_multi_seg, Y0_seg, X0_seg
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

def test_nn(X_seg: list,Y_seg: list, maxNeurons: int, maxLayers: int,
            alpha: float = 2, reg_factor: float = 0.5, **kargs) -> list:
    archs = []
    yields = []
    arch_yields = {}
    for numLayers in range(1,maxLayers+1):
        allArchs = all_comb(numLayers,maxNeurons)
        for arch in allArchs:
            try:
                w = neural_net_param(X_seg[0],Y_seg[0], numLayers, arch, alpha,
                                     reg_factor, **kargs)[0]
                prob_Y = nn_prediction(X_seg[1],w)[-1]
                pred_Y = (prob_Y > 0.5) * (1)
                rightRatio = np.mean((pred_Y == Y_multi_seg[1]) * (1),
                                     axis = (0,1))
                archs.append(arch)
                yields.append(rightRatio)
                print(arch, ' is OK')
            except:
                print(arch,' is not OK')
        arch_yields = [(archs[i],yields[i]) for i in range(len(archs))]
    return arch_yields, archs, yields

def test_logreg_param(X_cross: list, Y_cross: list, w: list) -> float:
    """
    Tests the parameters of a multi-class logistic regression model for 
    the percentage of success over a cross-evaluation set.
    """
    prob = [0] * len(X_cross)
    f1 = lambda x: logreg_prob(x,w)
    f2 = lambda x: np.concatenate([x[i] for i in range(len(x))], axis = 1)
    prob = f2(f1(X_cross))
    return prob

#w_multi = neural_net_param(X_seg[0],Y_multi_seg[0], 3,[3,3,3], alpha = 4)[0]
#w = neural_net_param(X0_seg[0],Y0_seg[0], 2, [2,2])[0]
#prob_multi_Y = nn_prediction(X_seg[1],w_multi)[-1]
#prob_Y = nn_prediction(X0_seg[1],w)[-1]
#pd.DataFrame(prob_Y)
#pd.DataFrame(prob_multi_Y)
#result, archs, yields = test_nn(X_seg,Y_multi_seg, maxNeurons = 3, maxLayers = 3,
#                 maxIteration = 10000)
#result = sorted(result, key = lambda x:-x[1])
w_log = logreg_param(X0_seg[0],Y0_seg[0])
prob = test_logreg_param(X0_seg[1],Y0_seg[1], w_log)