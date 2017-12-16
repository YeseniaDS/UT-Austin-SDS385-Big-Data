# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 12:49:28 2017

@author: ashutosh
"""

import pandas as pd
import numpy as np
import sklearn
from sklearn import decomposition as skd
import matplotlib.pyplot as plt
from sklearn import preprocessing


## this code imports MNIST Dataset
## Does PCA and selects 1st 255 columns , adds intercept
## converts labels to on hot encoded vectors
def datapre():
    data = pd.read_csv("mnist_train.csv")
    data.head(5)
    X_tp1 = data.iloc[:,1:].values
    X_tp2 = X_tp1[21]
    X_tp = np.vstack((X_tp1,X_tp2))
    Y_tp1 = data.iloc[:,0].values
    Y_tp = np.append(Y_tp1,Y_tp1[21])
    pca = skd.PCA()
    fit = pca.fit_transform(X_tp)
    inte = np.ones((60000,1))

## from the plot, the variance flatens out after 200 predictors, so we can 
## get rid of the rest of them
    X = preprocessing.scale(fit[:,0:255])
    X = np.hstack((inte,X))
    Y =np.eye(10)[Y_tp]
#del data, X_tp, Y_tp , inte

    data = pd.read_csv("mnist_test.csv")
    data.head(5)
    X_tp1 = data.iloc[:,1:].values
    X_tp2 = X_tp1[21]
    X_tp = np.vstack((X_tp1,X_tp2))
    Y_tp1 = data.iloc[:,0].values
    Y_tp = np.append(Y_tp1,Y_tp1[21])
    pca = skd.PCA()
    fit = pca.fit_transform(X_tp)
    
## from the plot, the variance flatens out after 200 predictors, so we can 
## get rid of the rest of them
    inte = np.ones((len(X_tp),1))
    X_test = preprocessing.scale(fit[:,0:255])
    X_test = np.hstack((inte,X_test))
    Y_test = np.array(Y_tp)
    del data, X_tp, Y_tp
    
    return X,Y,X_test,Y_test