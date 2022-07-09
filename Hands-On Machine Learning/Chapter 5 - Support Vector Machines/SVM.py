# -*- coding: utf-8 -*-
"""
Created on Sun May  2 15:43:48 2021

@author: filip

Chapter 5 - Support Vector Machine

"""

#SVM Linear example

import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
X = iris["data"][:,(2,3)] #petal lenght, petal width
y = (iris["target"] == 2).astype(np.float64)

svm_clf = Pipeline([("scaler",StandardScaler()),("linear_svc", LinearSVC(C=1, loss = "hinge"))])
svm_clf.fit(X,y)

svm_clf.predict([[5.5,1.7]])


#Non linear SVM

from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

X,y = make_moons(n_samples=100, noise = 0.15)
polynomial_svm_clf = Pipeline([("poly_features",PolynomialFeatures(degree=3)),
                               ("scaler", StandardScaler()),
                               ("svm_clf", LinearSVC(C=10, loss="hinge"))])

polynomial_svm_clf.fit(X,y)

import matplotlib.pyplot as plt
plt.scatter(X[:,0],X[:,1])


#Polynomial Kernel

from sklearn.svm import SVC

poly_kernel_svm_clf = Pipeline([("scaler",StandardScaler()),
                                ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
                                ])
poly_kernel_svm_clf.fit(X,y)

#Guassian RBF Kernel
rbf_kernel_svm_clf = Pipeline([("scaler",StandardScaler()),
                               ("svm_clf",SVC(kernel="rbf", gamma=5, C=0.001))
                               ])

rbf_kernel_svm_clf.fit(X,y)

#SVM Regression

from sklearn.svm import LinearSVR

svm_reg = LinearSVR(epsilon=1.5)
svm_reg.fit(X,y)

from sklearn.svm import SVR

svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=.1)
svm_poly_reg.fit(X,y)


