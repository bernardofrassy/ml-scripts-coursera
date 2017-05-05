#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 15:17:54 2017

@author: bernardoalencar
"""
import numpy as np

def k_means_clustering(dataSet: np.matrix, numClusters: int,
                       numTrials: int) -> np.matrix:
    """
    Applies K-means clustering method to a given dataset.
    """
    try:
        dataSet = np.matrix(dataSet)
    except:
        raise TypeError('dataSet cannot be converted into Matrix')
    n = dataSet.shape[0]
    lowerCost = None
    bestCord = None
    for i in range(numTrials):
        converge = False
        # Defining random initial K clusters from example:
        randExamples = [np.random.choice(range(n),numClusters, replace = False)]
        clusterCords = dataSet[randExamples]
        while converge == False:
            # Calculate closer clusts:
            dist = distance_to_clusters(dataSet, clusterCords)
            closerClust = (np.min(dist, axis = 1) == dist) * 1
            # Calculating new cluster coordenates:
            f1_cord = lambda x,y: np.sum(np.multiply(x,y),
                                         axis = 0)/np.sum(y, axis = 0)
            for i in range(closerClust.shape[1]):
                if i == 0:
                    newCords = f1_cord(dataSet, closerClust[:,i])
                else:
                    newCords = np.concatenate((newCords,
                                               f1_cord(dataSet, closerClust[:,i])),
                                             axis = 0)
            if (newCords == clusterCords).all() == True:
                converge = True 
            newCords = remove_zero_clusts(closerClust, newCords)
            newCords = remove_rep_clusts(newCords)
            clusterCords = newCords
        dist = distance_to_clusters(dataSet, clusterCords)
        closerClust = (np.min(dist, axis = 1) == dist) * 1
        cost = np.sum(np.min(dist, axis = 1))
        if lowerCost == None:
            lowerCost = cost
            bestCord = clusterCords
        if cost < lowerCost:
            bestCord = clusterCords
            lowerCost = cost
    return bestCord, lowerCost

def distance_to_clusters(dataSet, clusterMatrix):
    numClusters = clusterMatrix.shape[0]
    for i in range(numClusters):
        f_dist = lambda x,y: np.sum(np.square(x - y), axis = 1)
        if i == 0:
            distClusters = f_dist(dataSet, clusterMatrix[i])
        else:
            currDist = f_dist(dataSet, clusterMatrix[i])
            distClusters = np.concatenate((distClusters, currDist), axis = 1)
    return distClusters

def remove_zero_clusts(closerClust, clusterCords) -> (np.matrix, np.matrix):
    # Remove zero-items clust:
        zeroClusts = list()
        for i in range(closerClust.shape[1]):
            if (closerClust[:,i] == 0).all(axis = 0):
                zeroClusts.append(i)
        nonZeroCords = np.delete(clusterCords, zeroClusts, axis = 0)
        return nonZeroCords
    
def remove_rep_clusts(clusterCords):
    removedCords = clusterCords.copy()
    repIndex = list()
    for i in range(removedCords.shape[0]):
        if (removedCords[i,:] in np.delete(removedCords,i, axis = 0)):
            removedCords[i,:] = 0
            repIndex.append(i)
    removedCords = np.delete(removedCords, repIndex, axis = 0)
    return removedCords
    
#X = np.matrix(range(6))
#X = np.concatenate((X, (X*2 - 3), X*8 + 5,X), axis = 0)
#k_right = np.matrix([30,30,30,30],[])
#k, cost = k_means_clustering(test_data,numClusters = 4, numTrials = 100)
print(k)