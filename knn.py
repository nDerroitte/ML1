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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# (Question 2)

from sklearn.model_selection import cross_val_score
import operator

FIXED_NB = 12

def q21(x,y):
    trainSample = (x[:1000,:], y[:1000])
    testSample = (x[1000:,:], y[1000:])
    for  i in (1,5,50,100,500):
        knn =  KNeighborsClassifier(n_neighbors=i)
        estimator = knn.fit(trainSample[0], trainSample[1])
        yPredicted = estimator.predict(testSample[0])
        print("Accuracy with {} neighbors is : {}. ".format(i, accuracy_score(testSample[1], yPredicted)))
        name = "boundaryKNN"+ str(i)
        title = "Distibution for n_neighbors = " + str(i)
        plot_boundary(name, estimator, testSample[0], testSample[1],title=title)
    return


def q22(x,y):
    optiNeighbors = 0
    accuracies = []
    nbNeighbor = list(range(1,501))
    print("Computing 10 cross fold-validation for each number of neighbors. It may take a few moments...")
    for i in nbNeighbor:
        knn = KNeighborsClassifier( n_neighbors=i)
        accuracies.append(np.mean(cross_val_score(knn,x,y,cv=10,n_jobs = -1)))

    optiNeighbors, optiAcc = max(enumerate(accuracies), key=operator.itemgetter(1))
    optiNeighbors +=1

    #Ploting
    plt.figure()
    plt.plot(nbNeighbor, accuracies,'b')
    plt.plot(optiNeighbors, optiAcc,'r.')
    plt.xlabel("Number of neighbors")
    plt.ylabel("Accuracy")
    plt.savefig("1OFCVAccByNeighbors1.svg")

    plt.figure()
    plt.plot(nbNeighbor[10:], accuracies[10:],'b')
    plt.plot(optiNeighbors, optiAcc,'r.')
    plt.xlabel("Number of neighbors")
    plt.ylabel("Accuracy")
    plt.savefig("1OFCVAccByNeighbors2.svg")

    return optiNeighbors, optiAcc

def plotq22():
    return
if __name__ == "__main__":

    x, y = make_unbalanced_dataset(3000, FIXED_NB)
    q21(x,y)
    optiNeighbors, optiAcc = q22(x,y)
    print("The Optimal number of Neighbors, after using the 10 cross fold-validation, is  {} with the accuracy of {}".format(optiNeighbors,optiAcc))
