#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 20:05:00 2017

@author: bernardoalencar
"""
import numpy as np
import sklearn.datasets
import DataPreProcess
from NeuralNetwork import neural_net_param, nn_prediction

iris = sklearn.datasets.load_iris()
X = DataPreProcess.data_std_shape(iris.data[:, :2])
Y = DataPreProcess.data_std_shape(iris.target)

Y0,Y1,Y2 = DataPreProcess.data_class_separation(Y,[0,1,2])
X0_seg, Y0_seg = DataPreProcess.data_seg(X,Y0)
X1_seg, Y1_seg = DataPreProcess.data_seg(X,Y1)
X2_seg, Y2_seg = DataPreProcess.data_seg(X,Y2)

w, grad,neuron_list = neural_net_param(X0_seg[0],Y0_seg[0],2,[3,3], alpha = 0.5, reg_factor = 0.5)
prob_y0 = nn_prediction(X0_seg[1],w)[-1]
pred_y0 = (prob_y0 > 0.5) * (1)
print(sum((pred_y0 == Y0_seg[1]) * (1))/Y0_seg[1].shape[0])

w, grad,neuron_list = neural_net_param(X1_seg[0],Y1_seg[0],2,[3,3], alpha = 0.5, reg_factor = 0.5)
prob_y0 = nn_prediction(X1_seg[1],w)[-1]
pred_y0 = (prob_y0 > 0.5) * (1)
print(sum((pred_y0 == Y1_seg[1]) * (1))/Y1_seg[1].shape[0])

w, grad,neuron_list = neural_net_param(X2_seg[0],Y2_seg[0],2,[3,3], alpha = 0.5, reg_factor = 0.5)
prob_y0 = nn_prediction(X2_seg[1],w)[-1]
pred_y0 = (prob_y0 > 0.5) * (1)
print(sum((pred_y0 == Y2_seg[1]) * (1))/Y2_seg[1].shape[0])