#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 14:32:49 2017

@author: bernardoalencar

Machine Learning Functions
"""
import numpy as np


class Network:
    """
    Network to be used as a machine learning tool.
    """
    def __init__(self, train_data: 'pd.DataFrame or similar',
                 train_target: 'pd.DataFrame or similar'):
        self.train_data = train_data
        self.train_target = train_target

    def set_cross_data(self, cross_data: 'pd.DataFrame or similar',
                       cross_target: 'pd.DataFrame or similar'):
        self.cross_data = cross_data
        self.cross_target = cross_target

    def neural_net_hyp(self, numLayers: int, numNeurons: 'int or list',
                       lambdaFactor: float=0) -> (list, list):
        """
        Calculates the weights and neurons value for a given neural network
        architecture.
        """
        try:
            numLayers = self.numLayers
            numNeurons = self.numNeurons
        except:
            raise AttributeError('Network architecture is not defined')
        try:
            X = np.matrix(self.train_data)
            X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        except:
            raise TypeError('dataSet cannot be converted into Matrix')
        try:
            numLayers = int(numLayers)
        except:
            raise TypeError('numLayers must be an integer or float that can' +
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
                raise ValueError('NumNeurons must be a single value or a' +
                                 'vector of size numLayers')
                return
        if len(numNeurons)-1 != numLayers:
            raise ValueError('NumNeurons must be a single value or a vector' +
                             ' of size numLayers')
        # Defining parameters initial values:
        thetaList = [0]*(numLayers + 1)
        neuronList = [0]*(numLayers + 1)
        zList = [0]*(numLayers + 1)
        # Defining theta values:
        for i in range(len(thetaList)):
            if i == 0:
                thetaList[i] = np.random.rand(X.shape[1], numNeurons[i])
            else:
                thetaList[i] = np.random.rand(numNeurons[i-1]+1, numNeurons[i])
        # Defining neuron and z values:
        for i in range(numLayers+1):
            if i == 0:
                zList[i] = X * thetaList[i]
            else:
                zList[i] = neuronList[i-1] * thetaList[i]
            neuronList[i] = 1/(1 + np.exp(-zList[i]))
            neuronList[i] = np.concatenate((np.ones((X.shape[0], 1)), X),
                                           axis=1)
            self.theta = thetaList
            self.neuron = neuronList
            return self.theta, self.neuron

    def forward_propagation(self, X: np.matrix, thetaList: list) -> list:
        """
        Performs forward propagation to calculate neuron values.
        """
        neuronList = [0]*(len(thetaList) + 1)
        zList = [0]*(len(thetaList) + 1)
        for i in range(len(thetaList) + 1):
            if i == 0:
                neuronList[i] = X
            else:
                zList[i-1] = neuronList[i-1] * thetaList[i-1]
                neuronList[i] = 1/(1 + np.exp(-zList[i-1]))
                if i < len(thetaList):
                    onesRow = np.ones((neuronList[i].shape[0], 1))
                    neuronList[i] = np.concatenate((onesRow, neuronList[i]),
                                                   axis=1)
        return neuronList

    def check_shape(a: list) -> None:
        """
        Prints shape of components inside the given argument.
        """
        for i in range(len(a)):
            print('For %s shape is: %s' % (i, a[i].shape))
        return

    def backpropagation(self, X: np.matrix, thetaList: list,
                        neuronList: list,
                        lambdaFactor: float=0) -> list:
        """
        Calculates the gradient using backpropagation algorithm for
        neural networks.
        """
        Y = self.train_target
        lenTheta = len(thetaList)
        numLayers = lenTheta - 1
        # Defining partial derivative to z:
        delta = [0] * lenTheta
        dg_z = [0] * lenTheta
        gradient = [0] * lenTheta
        for i in range(lenTheta):
            dg_z[i] = np.multiply(neuronList[i+1], (1-neuronList[i+1]))
        # Calculating deltas:
        for i in range(numLayers+1)[::-1]:
            if i == (numLayers):
                delta[i] = neuronList[i+1] - Y
            else:
                delta[i] = np.multiply((delta[i + 1] *
                                        thetaList[i + 1][1:, :].T),
                                       dg_z[i][:, 1:])
        for i in range(lenTheta):
            gradient[i] = np.zeros(thetaList[i].shape)
        for i in range(numLayers + 1):
            gradient[i] = (neuronList[i].T * delta[i])/X.shape[0]
            gradient[i] = gradient[i] + lambdaFactor * thetaList[i]
        return gradient

    def neural_net_cost(self, X: np.matrix, thetaList: list,
                        neuronList: list, lambdaFactor: float=0) -> float:
        """
        Calculates the cross entropy for a given neural network.
        """
        Y = self.train_target
        hyp = neuronList[-1]
        m = X.shape[0]

        def fSquare(x): return np.sum(np.sum(np.square(x)))
        thetaSum = [fSquare(thetaList[i]) for i in range(len(thetaList))]
        thetaSum = np.sum(thetaSum)

        def fMult(x, y): return np.sum(np.multiply(x, y), axis=(0, 1))
        cost = (-fMult(Y, np.log(hyp)) - fMult((1 - Y), np.log(1 - hyp)))/m
        cost = cost + lambdaFactor/m * thetaSum
        cost = np.sum(cost)
        return cost

    def grad_descent(self, X: np.matrix, costFunction: 'function',
                     gradFunction: 'function',
                     w_initial: np.matrix, alpha: float=10**(-2),
                     lambdaFactor: float=0, maxIteration: int=100000,
                     printIteration: bool=False, momentum: bool=False):
        """
        Performs gradient descent for a given neural network.
        """
        # Initial guess (all zeros):
        w = w_initial
        # Apply gradient descent:
        count = 0
        countIncrease = 0
        error = 0.1
        eps = 1 - 10**(-6)
        velo = [0] * len(w)
        costOld = 10
        wNew = [0] * len(w)
        while ((error > 10**(-20)) and (count < maxIteration)):
            neuronList = self.forward_propagation(X, w)
            cost = costFunction(X, w, neuronList, lambdaFactor=lambdaFactor)
            grad = gradFunction(X, w, neuronList, lambdaFactor=lambdaFactor)
            if printIteration:
                print('Count ', count, 'Cost: ', cost)
            if (not momentum) or (count == 0):
                for i in range(len(w)):
                    wNew[i] = w[i] - grad[i] * alpha
                    velo[i] = wNew[i] - w[i]
            if (momentum) and (count != 0):
                for i in range(len(w)):
                    wNew[i] = w[i] + (eps * velo[i] - (grad[i] * alpha))
                    velo[i] = wNew[i] - w[i]
            # In case the cost Function increases:
            if cost > costOld:
                countIncrease += 1
                if countIncrease == (maxIteration * (10**(-4)) + 1):
                    print('Cost function is increasing too frequently.' +
                          ' Alpha reduced.')
                    alpha = 0.8 * alpha
                    countIncrease = 0
            error = abs((cost - costOld)/cost)
            w = wNew
            costOld = cost
            count += 1
        if printIteration:
            print(error, count)
        self.theta = w
        self.cost = cost
        self.neuron = neuronList

    def grad_check(self, X: np.matrix, theta: list,
                   neuron: list, grad: list) -> bool:
        """
        Checks the if computed gradient is the same as the expected gradient.
        """
        eps = 1
        expGrad = 0
        diffGrad = True
        for i in range(len(theta)):
            for dest in range(theta[i].shape[0]):
                for weight in range(theta[i].shape[1]):
                    theta[i][dest, weight] += eps
                    upperCost = self.neural_net_cost(X, theta,
                                                     neuron)/(2 * eps)
                    theta[i][dest, weight] -= (2 * eps)
                    lowerCost = self.neural_net_cost(X, theta,
                                                     neuron)/(2 * eps)
                    expGrad = (upperCost - lowerCost)
                    theta[i][dest, weight] += eps
                    if abs(expGrad - grad[i][dest, weight]) > 10**(-5):
                            diffGrad = False
        return diffGrad

    def train(self, numLayers: int, numNeurons: int,
              alpha: float=2, lambdaFactor: float=0,
              **kargs) -> (np.matrix, np.matrix, list):
        """
        Trains the weights for a given neural network architecture.
        """
        try:
            X = np.matrix(self.train_data)
            X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
            Y = np.matrix(self.train_target)
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
        # Save given configuration as instance attributes:
        self.numLayers = numLayers
        self.numNeurons = numNeurons
        self.lambdaFactor = lambdaFactor

        # Defining parameters initial values:
        thetaList = [0]*(numLayers + 1)

        # Random inital theta_values:
        for i in range(len(thetaList)):
            if i == 0:
                thetaList[i] = np.random.rand(X.shape[1], numNeurons[i])
            else:
                thetaList[i] = np.random.rand(numNeurons[i-1]+1, numNeurons[i])
        self.costFunction = self.neural_net_cost
        self.grad_descent(X, self.costFunction, self.backpropagation,
                          w_initial=thetaList, alpha=alpha, **kargs)

    def predict_targets(self):
        try:
            thetaList = self.theta
        except:
            raise AttributeError('Theta attribute not defined')
        try:
            X = self.cross_data
        except:
            raise AttributeError('Cross data set not defined')
        try:
            X = np.matrix(X)
            X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        except:
            raise ValueError('cross_data attribute cannot be ' +
                             'converted into Matrix')
        neuronList = [0]*(len(thetaList) + 1)
        zList = [0]*(len(thetaList) + 1)
        for i in range(len(thetaList) + 1):
            if i == 0:
                neuronList[i] = X
            else:
                zList[i-1] = neuronList[i-1] * thetaList[i-1]
                neuronList[i] = 1/(1 + np.exp(-zList[i-1]))
                if i < len(thetaList):
                    onesRow = np.ones((neuronList[i].shape[0], 1))
                    neuronList[i] = np.concatenate((onesRow, neuronList[i]),
                                                   axis=1)
        self.prediction_prob = neuronList[-1]
        prob = self.prediction_prob
        if prob.shape[1] > 1:
            pred = ((prob == np.amax(prob, axis=1)) * 1)
            numWrong = -np.sum((pred == self.cross_target) * 1 - 1,
                               axis=(0, 1))/2
        else:
            pred = ((prob > 0.5) * 1)
            numWrong = -np.sum((pred == self.cross_target) * 1 - 1,
                               axis=(0, 1))
        rightRatio = float(1 - numWrong/prob.shape[0])
        self.prediction = pred
        self.success_ratio = rightRatio
