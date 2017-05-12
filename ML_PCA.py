#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:09:40 2017

@author: bernardoalencar
"""
import numpy as np
def mean_normalize(X: np.matrix) -> np.matrix:
    mean = np.mean(X, axis = 0)
    return X - mean

def scale_max(X: np.matrix) -> np.matrix:
    max_value = np.max(X, axis = 0)
    return X / max_value


def PCA(X: np.matrix, dim: int) -> np.matrix:
    Xnormed = mean_normalize(X)
    Xnormed = scale_max(Xnormed)
    covMatrix = np.cov(Xnormed.T)
    eigVectors = np.linalg.eig(covMatrix)[1]
    z = eigVectors[:,:dim]
    Xreduced = np.dot(X,z)
    return Xreduced