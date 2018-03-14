"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Only py3 / so that 2 / 3 = 0.66..
from __future__ import division
# Only py3 string encoding
from __future__ import unicode_literals
# Only py3 print
from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt

from plot import plot_boundary
from data import make_balanced_dataset, make_unbalanced_dataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split


# (Question 1)
from sklearn import tree
import operator

FIXED_NB = (12, 35, 48 , 68, 95)
depth = (None, 1 , 2 , 4, 6, 8)
global DScount


def overfittingCheck(trainSample,best_depth,bestAcc):
    accBestTS = predDT(trainSample,trainSample,best_depth)

    for i in depth:
        if i!= None and i <= best_depth:
            continue
        currentAccTS = predDT(trainSample,trainSample,i)
        if(currentAccTS> accBestTS):
            print("The tree (dataset 0) of depth {} is overfitting ({} {})".format(i,accBestTS,currentAccTS))
    return
def underfittingCheck(trainSample):
    for i in depth:
        currentAccTS = predDT(trainSample,trainSample,i)
        if(currentAccTS <= 0.90):
            print("The tree (dataset 0) of depth {} is underfitting ({})".format(i,currentAccTS))
    return
def predDT(trainSample,testSample,max_depth=None):
    dt = DecisionTreeClassifier(max_depth=max_depth,random_state = 42)
    estimator = dt.fit(trainSample[0], trainSample[1])
    yPredicted = estimator.predict(testSample[0])
    acc = accuracy_score(testSample[1], yPredicted)
    return acc

def predPlotDT(trainSample,testSample,max_depth=None,plot=False):
    #Generation of the training and test datasets
    global DScount

    #Computing the acc. Not using predDT function because we need the dt in order to plot
    dt = DecisionTreeClassifier(max_depth=max_depth,random_state = 42)
    estimator = dt.fit(trainSample[0], trainSample[1])
    yPredicted = estimator.predict(testSample[0])
    acc = accuracy_score(testSample[1], yPredicted)

    if max_depth!= None:
        print("The accuracy of dataset {} with max depth of {} is {}. ".format(DScount,max_depth,acc))
    else :
        print("The accuracy of dataset {} without max depth is {}. ".format(DScount,acc))

    if(plot):
        print("Saving the file.")
        name = "boundaryDT"+str(max_depth)
        title = "Distibution for max depth tree of " + str(max_depth) if max_depth!= None else "Distibution for tree without max depth"
        plot_boundary(name, estimator, testSample[0], testSample[1],title=title)
        name = "boundaryDTLS"+str(max_depth)
        title = "Distibution for max depth tree of " + str(max_depth) + " for the learning sample"  if max_depth!= None else "Distibution for tree without max depth for the learning sample"
        plot_boundary(name, estimator, trainSample[0], trainSample[1],title=title)
        nameTree = "Tree" + str(max_depth) + ".dot"
        tree.export_graphviz(dt, out_file=nameTree)

    return acc

def genAccList(trainSample,testSample,plot=False):
    accList = []
    for i in depth:
        accList.append(predPlotDT(trainSample,testSample,i,plot=plot))

    print("--")

    return accList

def makeDataset (FIXED_NB_INDEX):
    x, y = make_unbalanced_dataset(3000, FIXED_NB[FIXED_NB_INDEX])
    trainSample = (x[:1000,:], y[:1000])
    testSample = (x[1000:,:], y[1000:])
    return (trainSample, testSample)


if __name__ == "__main__":
    #Q1
    global DScount
    DScount = 0
    accListByDS = []
    avgAccList  = []
    stdAccList  = []

    trainSampleDS0, testSampleDS0 = makeDataset(0)
    accListByDS.append(genAccList(trainSampleDS0, testSampleDS0,True))
    DScount +=1

    #Q2
    for i in range (1,5):
        trainSampleTmp,testSampleTmp = makeDataset(i)
        accListByDS.append(genAccList(trainSampleTmp,testSampleTmp))
        DScount+=1

    accListByDS = np.array(accListByDS)
    for j in range(5):
        avgAccList.append(np.mean(accListByDS[:,j]))
        stdAccList.append(np.std(accListByDS[:,j]))
    avgAccList = np.around(avgAccList,decimals=5)
    stdAccList = np.around(stdAccList,decimals=5)
    totalAccAvg = np.mean(avgAccList)
    print("Average accuracy by depth : {}\n(Population) Standard deviation by depth : {}\nTotal average accuracy : {}".format(avgAccList,stdAccList,totalAccAvg))
    print("--")

    #Checking for overfitting
    #We take the index of the best tree of our first dataset and compute the error on the learning sample
    indexBestAccDS0, bestAccDS0 = max(enumerate(accListByDS[0]), key=operator.itemgetter(1))
    bestDepthDS0 = depth[indexBestAccDS0]
    overfittingCheck(trainSampleDS0,bestDepthDS0,bestAccDS0)
    underfittingCheck(trainSampleDS0)
