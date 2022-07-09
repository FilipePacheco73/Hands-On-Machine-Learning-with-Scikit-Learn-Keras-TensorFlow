# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 14:11:00 2021

@author: filip

Identificação de padrões com números

"""

#libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

#Download do dataset - disponível na Web
mnist = fetch_openml('mnist_784',version=1)
mnist.keys()

#Import dataset

X,y = mnist["data"], mnist["target"]

X = np.array(X)
y = np.array(y)

some_digit = X[0]
some_digit_image = some_digit.reshape(28,28)

plt.imshow(some_digit_image, cmap="binary")
plt.axis("off")

y[0]

y = y.astype(np.uint8)

#Train the Classifier
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

sgd_clf = SGDClassifier(random_state = 42)
sgd_clf.fit(X_train, y_train_5)

sgd_clf.predict([some_digit])

#Implementing cross-validation to measure the models performance

skfolds = StratifiedKFold(n_splits=3)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train[test_index]
    
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct/len(y_pred))
    
cross_val_score(sgd_clf,X_train,y_train_5,cv=3,scoring="accuracy")

#Confusion Matrix - another way to measure the performance

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

confusion_matrix(y_train_5, y_train_pred) # shows the falses positive and negative

precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)

f1_score(y_train_5, y_train_pred)

#Precision/Recall trade-off

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,method="decision_function")

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

def plot_precision_recall_vs_threshold(precisions,recalls,thresholds):
    plt.plot(thresholds, precisions[:-1],"b--",label="Precision")
    plt.plot(thresholds, recalls[:-1],"g-",label="Recall")
    
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.plot(precisions,recalls)

threshold_90_precision = thresholds[np.argmax(precisions >= .9)]

y_train_pred_90 = (y_scores >= threshold_90_precision)

precision_score(y_train_5, y_train_pred_90)
recall_score(y_train_5, y_train_pred_90)

#ROC curve

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

def plot_roc_curve(fpr, tpr, label = None):
    plt.plot(fpr, tpr, linewidth=2, label = label)
    plt.plot([0,1],[0,1], "k--")

plot_roc_curve(fpr, tpr)

#Random Forest Classifier 

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)

y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")

y_scores_forest = y_probas_forest[:,1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")

"""
#Multiclass Classification

from sklearn.svm import SVC
svm_clf = SVC()

svm_clf.fit(X_train, y_train) # OvR - One-versus-the-rest
svm_clf.predict([some_digit])

some_digit_scores = svm_clf.decision_function([some_digit])

np.argmax(some_digit_scores)

svm_clf.classes_

from sklearn.multiclass import OneVsRestClassifier
ovr_clf = OneVsRestClassifier(SVC())
ovr_clf.fit(X_train, y_train)
ovr_clf.predict([some_digit])

sgd_clf.fit(X_train, y_train) # Training multiclass with SGD
sgd_clf.predict([some_digit])

sgd_clf.decision_function([some_digit])

cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring = "accuracy")

#Error Analysis

y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)

plt.matshow(conf_mx, cmap=plt.cm.gray)

row_sums = conf_mx.sum(axis =1 , keepdims=True)

norm_conf_mx = conf_mx / row_sums

np.fill_diagonal(norm_conf,mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
"""