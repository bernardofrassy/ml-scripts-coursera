#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 16:11:17 2017

@author: bernardoalencar
Functions for data pre-processing.
"""
import numpy as np
import random as rd


def lr_data_creation(n_samples: int, n_features: int = 1, n_targets: int = 1,
                     random_state=0, noise=35, **kargs
                     ) -> (np.ndarray, np.ndarray):
    """
    Creates data to test a linear regression.
    """
    from sklearn.datasets.samples_generator import make_regression
    X, Y = make_regression(n_samples=n_samples, n_features=n_features,
                           n_targets=n_targets,
                           n_informative=n_features//2,
                           random_state=random_state, noise=noise,
                           **kargs)
    Y = data_std_shape(Y)
    return X, Y


def data_to_class(data: 'np array-like', numClasses: int) -> 'np array-like':
    """
    Converts a linear array to one with a defined number of classes.
    """
    data = normalize_data(data)
    dataClass = np.copy(data)
    dataClass[:] = 0
    for i in range(0, numClasses):
        upperMask = (data < np.percentile(data, (i+1) * 100/numClasses))
        lowerMask = (data > np.percentile(data, (i) * 100/numClasses))
        mask = np.multiply(upperMask, lowerMask)
        dataClass[mask] = i
    return dataClass


def logreg_data_creation(n_samples: int, n_features: int = 1,
                         n_classes: int = 2, random_state=0, noise=35,
                         **kargs) -> (np.ndarray, np.ndarray):
    """
    Creates data to test a logistic regression.
    """
    from sklearn.datasets.samples_generator import make_regression
    X, Y = make_regression(n_samples=n_samples, n_features=n_features,
                           n_targets=1,
                           n_informative=n_features//2,
                           random_state=random_state, noise=noise,
                           **kargs)
    Y = data_to_class(Y, n_classes)
    return X, Y


def data_seg(X: np.ndarray, Y: np.ndarray,
             cut_values: list = [0.6, 0.2, 0.2]) -> (list, list):
    """
    Segments data into 3 sets, according to given percentages.
    """
    dataLength = X.shape[0]
    seg = np.array([rd.random() for i in range(dataLength)])
    trainMask = (seg <= cut_values[0])
    crossMask = (seg <= sum(cut_values[:2])) * (seg > cut_values[0])
    testMask = (seg > sum(cut_values[:2]))
    X_seg = [X[trainMask], X[crossMask], X[testMask]]
    Y_seg = [Y[trainMask], Y[crossMask], Y[testMask]]
    return X_seg, Y_seg


def normalize_data(data: 'np.matrix, pd.DataFrame or similar'
                   ) -> np.matrix:
    """
    Normalizes a given dataset.
    """
    dataCopy = np.matrix(data)
    if len(dataCopy.shape) == 1:
        data = dataCopy.reshape((len(data), 1))
    for i in range(dataCopy.shape[1]):
        if np.std(dataCopy[:, i]) != 0:
            dataCopy[:, i] = (dataCopy[:, i] - np.average(dataCopy[:, i])
                              )/np.std(dataCopy[:, i])
    return dataCopy


def data_class_separation(data: np.array, class_list: list) -> list:
    """
    Separates the data according to a list of classes.
    The output says if the data point is contained or not in each class.
    """
    data_sep = [np.copy(data)] * len(class_list)
    for i in range(len(class_list)):
        data_sep[i] = 1 * (data_sep[i] == class_list[i])
    return data_sep


def data_std_shape(data) -> np.ndarray:
    """
    Converts the data for the standard used in the programs.
    """
    if type(data) != np.ndarray:
        data = np.array(data)
    if len(data.shape) == 1:
        data = data.reshape((len(data), 1))
    if data.shape[0] < data.shape[1]:
        print('Number of rows higher than number of columns.',
              'Check for possible mistake.')
    return data
