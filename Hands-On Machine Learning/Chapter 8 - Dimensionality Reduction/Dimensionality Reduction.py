# -*- coding: utf-8 -*-
"""
Created on Sat May 22 13:43:55 2021

@author: filip

Chapter 8 - Dimensionality Reduction

"""

# Principal Component Analysis - PCA

import numpy as np
from sklearn.datasets import fetch_openml

#Download do dataset - disponível na Web
mnist = fetch_openml('mnist_784',version=1)
mnist.keys()

#Import dataset

X,y = mnist["data"], mnist["target"]

X = np.array(X)
y = np.array(y)

#Train the Classifier
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

X_centered = X - X.mean(axis=0)

U, s, Vt = np.linalg.svd(X_centered)

c1 = Vt.T[:,0]
c2 = Vt.T[:,1]

W2 = Vt.T[:,:2]
X2D = X_centered.dot(W2)

from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
X2D = pca.fit_transform(X)

pca.explained_variance_ratio_

#Chossing the right number of Dimensions

pca = PCA()
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= .95) + 1

pca = PCA(n_components = .95)
X_reduced = pca.fit_transform(X_train)

pca = PCA(n_components = .95)
X_reduced = pca.fit_transform(X_train)

#PCA for Compression

pca = PCA(n_components = 154)
X_reduced = pca.fit_transform(X_train)
X_recovered = pca.inverse_transform(X_reduced)

#Randomized PCA

rnd_pca = PCA(n_components = 154, svd_solver = "randomized")
X_reduced = rnd_pca.fit_transform(X_train)

#Incremental PCA

from sklearn.decomposition import IncrementalPCA

n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_train, n_batches):
    inc_pca.partial_fit(X_batch)
    
X_reduced = inc_pca.transform(X_train)

#Kernel PCA

from sklearn.decomposition import KernelPCA

rbf_pca = KernelPCA(n_components = 2, kernel = "rbf", gamma = .04)
X_reduced = rbf_pca.fit_transform(X)

