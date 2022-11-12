# -*- coding: utf-8 -*-
"""
Created on Sat May 15 15:15:53 2021

@author: Filipe Pacheco

Hands-On Machine Learning

Chapter 6 - Decision Tree

"""

#Packages

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz

# Decision Tree Classifier

iris = load_iris()

X = iris.data[:,2:]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 42)

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X_train,y_train)

# Export the graphics of the decision tree

export_graphviz(
    tree_clf,
    #out_file=image_path("iris_tree.dot"),
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    rounded=True,
    filled=True)

tree_clf.predict_proba([[5,1.5]])

tree_clf.predict([[5,1.5]])

accuracy_score(y_test,tree_clf.predict(X_test))


# Decision Tree Regression

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

tree_reg = DecisionTreeRegressor(max_depth=5)

tree_reg.fit(X_train,y_train)

r2_score(y_test,tree_reg.predict(X_test))
