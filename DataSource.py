#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 19:12:42 2017

@author: bernardoalencar
Contains the Data used to test scripts.
"""
import numpy as np
import sklearn.datasets
import DataPreProcess

iris = sklearn.datasets.load_iris()
X = DataPreProcess.data_std_shape(iris.data[:, :2])
Y = DataPreProcess.data_std_shape(iris.target)

Y0,Y1,Y2 = DataPreProcess.data_class_separation(Y,[0,1,2])
X0_seg, Y0_seg = DataPreProcess.data_seg(X,Y0)
X1_seg, Y1_seg = DataPreProcess.data_seg(X,Y1)
X2_seg, Y2_seg = DataPreProcess.data_seg(X,Y2)
Y_multi = np.concatenate((Y0,Y1,Y2), axis = 1)
X_seg, Y_multi_seg = DataPreProcess.data_seg(X,Y_multi)
