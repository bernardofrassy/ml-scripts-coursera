#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 12:02:37 2017

@author: bernardoalencar
"""
import numpy as np
import pandas as pd
import re
import NeuralNetwork as nn
import DataPreProcess as dpp
import ML_LogisticsRegression as lr
import TestNN as test

with open('train.csv','r') as file:
    count = 0
    data = np.zeros((1,12))
    for line in file:
        if count == 0:
            variableNames = line.rsplit( sep = ',')
        elif count == 1:
            data = np.array(line.rsplit( sep = ',')).reshape((1,13))
        else:
            lineArray = np.array(line.rsplit( sep = ',')).reshape((1,13))
            data = np.concatenate([data, lineArray], axis = 0)
        count += 1

# Fixing surnames:
variableNames.insert(4, 'Surnames')
df = pd.DataFrame(data)

# Renaming columns:
dictNames = {int(i) : variableNames[i] for i in range(13)}
dictNames[12] = 'Embarked'
df = df.rename(columns = dictNames)

# Fixing data format:
# Converting Embarked port to integers & None:
df['Embarked'] = df['Embarked'].apply(lambda x: x[0])
df['Embarked'][df['Embarked'] == '\n'] = None
portNames = list(set(df['Embarked']))
portNames.remove(None)
portDict = {portNames[i] : int(i) for i in range(len(portNames))}
portDict.update({None : 'nan'})
df['Embarked'] = df['Embarked'].apply(lambda x: portDict[x])
df['Embarked'][ df['Embarked'] == 'nan'] = None

# Converting Sex and Age to intergers & None:
df['Sex'] = (df['Sex'] == 'male') * 1
df.loc[df['Age'] == '', 'Age'] = None

# Removing characters from ticket names:
f1 = lambda x: re.findall("[0-9]*$", x)[0]
df['Ticket'] = df['Ticket'].apply(f1)
maxTicket = max(df['Ticket'])

# Selecting Data to aply Neural Netowrk:
data_nn = df.loc[:,('Survived','PassengerId', 'Pclass', 'Sex',	 'Age', 'SibSp', 'Parch',
             'Ticket', 'Fare', 'Embarked')]
data_nn = data_nn.apply(pd.to_numeric, axis = 1)
data_nn = data_nn.dropna()
data_nn['Ticket'] = data_nn['Ticket']/max(data_nn['Ticket'])
normColumns = ['PassengerId','Age','Fare']
data_nn.loc[:,normColumns] = dpp.normalize_data(data_nn.loc[:,normColumns])
data_nn['Pclass'] = data_nn['Pclass'] - 2
data_nn['Embarked'] = data_nn['Embarked'] - 1
Y = dpp.data_std_shape(data_nn['Survived'])
X = dpp.data_std_shape(data_nn.iloc[:,1:])

X_seg, Y_seg = dpp.data_seg(X,Y, [0.75,0.25,0])
X_seg = X_seg[:2]
Y_seg = Y_seg[:2]

# Applying NN:
net1 = nn.Network(X_seg[0], Y_seg[0])
net1.train(2, [2,2], alpha = 20, printIteration = True, momentum = True)
net1.set_cross_data(X_seg[1], Y_seg[1])
net1.predict_targets()
sum(net1.prediction)
net1.success_ratio

# Applying Logistics Regression:
theta = lr.logreg_param(X_seg[0], Y_seg[0])
sucess_ratio, prob = test.test_param(X_seg[1], Y_seg[1], theta,
                                     lr.logreg_prob)
print(sucess_ratio)