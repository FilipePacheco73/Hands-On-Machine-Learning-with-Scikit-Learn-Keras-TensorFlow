# -*- coding: utf-8 -*-
"""
Created on Sat May 15 15:15:53 2021

@author: filip

Chapter 6 - Decision Tree

"""

# Decision Tree Classifier

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()

X = iris.data[:,2:]
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X,y)

from sklearn.tree import export_graphviz

export_graphviz(
    tree_clf,
    out_file=image_path("iris_tree.dot"),
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    rounded=True,
    filled=True)

tree_clf.predict_proba([[5,1.5]])

tree_clf.predict([[5,1.5]])

# Decision Tree Regression

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X,y)
