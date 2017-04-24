#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 13:25:55 2017

@author: bernardoalencar
"""
from NeuralNetwork import neural_net_param

def all_comb(n: int, k: int) -> list:
    """
    Create all possible combinations for \'n\' items with \'k\'
    possible classes.
    """
    numComb = n ** k
    allComb = list()
    currComb = list()
    for value in range(0,numComb):
        currComb = [(value // k ** i) % (k) + 1 for i in range(0,n)]
        allComb.append(currComb)        
    return allComb

def test_nn(X,Y, maxNeurons: int, maxLayers: int, alpha: float = 1,
            reg_factor: float = 0.5) -> None:
    for numLayers in range(maxLayers):
        allArchs = all_comb(numLayers,maxNeurons)
        for arch in allArchs:
            try:
                neural_net_param(X,Y,numLayers,arch, alpha, reg_factor)
                print(arch,'is OK')
            except:
                print(arch,'is not OK')
    return

b = all_comb(3,4)
print(b)