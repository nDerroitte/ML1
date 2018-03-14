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

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin


from data import make_balanced_dataset, make_unbalanced_dataset
from plot import plot_boundary
from sklearn.metrics import confusion_matrix, accuracy_score

import time
import operator
from matplotlib import pyplot as plt

FIXED_NB = (14, 35, 48 , 68, 95)


class LogisticRegressionClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_iter=10, learning_rate=1):
        self.n_iter = n_iter
        self.learning_rate = learning_rate


    def fit(self, X, y):
        """Fit a logistic regression models on (X, y)

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X = np.asarray(X, dtype=np.float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")
        n_instances, n_features = X.shape

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        n_classes = len(np.unique(y))
        if n_classes != 2:
            raise ValueError("This class is only dealing with binary "
                             "classification problems")

        self.n_instances_gen = n_instances
        self.n_features_gen = n_features

        theta = self.initTheta(n_features)
        self.bestTheta = theta

        bestAcc = 0

        #Compute Optimal Theta

        for i in range (self.n_iter):
            #UpdateTheta
            theta  = self.updateTheta(theta,X,y)

            iterY = np.around(self.computeY(X,theta),decimals= 0)
            iterAcc= accuracy_score(y,iterY)
            if iterAcc > bestAcc:
                self.bestTheta =  theta

        return self

    def initTheta(self,n_features):
        w0 = 5

        theta = []
        theta.append(w0)

        for i in range (n_features):
            theta.append(0)
        return theta

    def updateTheta(self,theta,X,y):
        newTheta = []
        gradLL = self.gradientLL(X,y,theta)
        for i in range (self.n_features_gen+1):
            newTheta.append(theta[i] - self.learning_rate * gradLL[0,i] )
        return newTheta

    def proba_calculation(self,X, theta):
        p =np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            p[i] = 1/(1+np.exp(-theta[0]-np.dot(theta[1:],X[i,:])))
        return p

    def gradientLL(self,X, y, theta):
        p = np.subtract(self.proba_calculation(X, theta), y)
        p = np.transpose(p)

        w = np.ones((self.n_instances_gen,1))
        np.append(w, X)

        v = np.zeros(w.shape[0])
        gradLL = np.zeros((1,self.n_features_gen+1))
        for i in range(self.n_instances_gen):
            v[i] = np.dot(p[i], w[i,:])
            gradLL = np.add(gradLL,v[i])
        for i in range(gradLL.shape[1]):
            gradLL[0,i] /= self.n_instances_gen
        return gradLL

    def computeY(self,X,theta):
        Y = self.proba_calculation(X,theta)
        return Y



    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """
        return  np.around(self.computeY(X,self.bestTheta),decimals=0)


    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        y = self.computeY(X,self.bestTheta)
        p = np.zeros((X.shape[0],2))
        for i in range(X.shape[0]):
            #print(i)
            p[i,0] =  1-y[i]
            p[i,1] =  y[i]
        return p

def findBestLR(trainSample,testSample):
    accList = []
    learning_rate = range(1,30)
    for i in learning_rate:
        lr = LogisticRegressionClassifier(learning_rate=i)
        lr.fit(trainSample[0],trainSample[1])
        yPredicted = lr.predict(testSample[0])
        accList.append(accuracy_score(testSample[1], yPredicted))

    optiLR, optiAcc = max(enumerate(accList), key=operator.itemgetter(1))
    optiLR +=1

    plt.figure()
    plt.plot(learning_rate, accList,'b')
    plt.plot(optiLR, optiAcc,'r.')
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.savefig("LROptiLR.png")
    print("The Optimal Learning rate is {}".format(optiLR))

    return optiLR

def findNbIter(trainSample,testSample,plot=False):
    nbIter = [1,10,20,50,100,200,500,1000]
    bestAcc = 0
    bestIter = 0
    for i in nbIter:
        start_time = time.time()
        lr = LogisticRegressionClassifier(n_iter=i)
        lr.fit(trainSample[0],trainSample[1])
        yPredicted = lr.predict(testSample[0])
        currentAcc = accuracy_score(testSample[1], yPredicted)
        if bestAcc < currentAcc :
            bestAcc = currentAcc
            bestIter = i
        print("Accuracy for {} iterations is {}. It took {} sec.".format(i,currentAcc, time.time() - start_time))
        if(plot):
            name = "boundaryLR" +str(i)
            title = "Distibution for " + str(i)+ "iterations."
            plot_boundary(name, lr, testSample[0], testSample[1],title=title)


    print("The Optimal number of iterations is {}".format(bestIter))

    return bestIter

def makeDataset (FIXED_NB_INDEX):
    x, y = make_unbalanced_dataset(3000, FIXED_NB[FIXED_NB_INDEX])
    trainSample = (x[:1000,:], y[:1000])
    testSample = (x[1000:,:], y[1000:])
    return (trainSample, testSample)

def predLR (n_iter, learning_rate, trainSample, testSample,plot=False):
    lr = LogisticRegressionClassifier(n_iter=n_iter,learning_rate=learning_rate)
    lr.fit(trainSample[0],trainSample[1])
    yPredicted = lr.predict(testSample[0])
    acc = accuracy_score(testSample[1], yPredicted)
    if(plot):
        name = "boundaryLR"
        title = "Distibution for " + str(n_iter)+ "iterations and a learning_rate of" +str(learning_rate)+"."
        plot_boundary(name, lr, testSample[0], testSample[1],title=title)
    return acc


if __name__ == "__main__":
    accListByDS = []
    avgAccList  = []
    LRList = []
    nbIterList = []

    #Comment this loop and set initial value for averageLR and averageNbIter to gain time.
    #print("Computing the best learning rate and the best nb of iterations")
    #for i in range (5):
    #    print("Dataset {}.".format(i))
    #    trainSample,testSample = makeDataset(i)
    #    if i ==0 :
    #        nbIterList.append(findNbIter(trainSample,testSample,plot= True))
    #    else :
    #        nbIterList.append(findNbIter(trainSample,testSample))
    #    LRList.append(findBestLR(trainSample,testSample))

    #averageLR = int(np.around(np.mean(LRList), decimals = 0))
    #averageNbIter = int(np.around(np.mean(nbIterList),decimals=0))
    print("Average optimal learning rate is {}.\nAverage optimal number of iterations is {}.".format(averageLR,averageNbIter))
    #End of computing the best LR and best nb of iters

    #Uncomment this line if you commented the previous loop
    averageLR = 15
    averageNbIter = 500

    for i in range (5):
        trainSample,testSample = makeDataset(i)
        if i ==0:
            avgAccList.append(predLR (averageNbIter,averageLR,trainSample,testSample,True))
        else :
            avgAccList.append(predLR (averageNbIter,averageLR,trainSample,testSample))
        print("The accuracy for dataset {} with LR = {} and {} iterations is {}".format(i,averageLR,averageNbIter,avgAccList[i]))

    totalAccAvg = np.mean(avgAccList)
    std = np.std(avgAccList)
    print("(Population) Standard deviation: {}\nTotal average accuracy : {}".format(std,totalAccAvg))
