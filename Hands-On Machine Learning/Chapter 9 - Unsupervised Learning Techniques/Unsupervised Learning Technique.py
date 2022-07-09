# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 10:39:57 2021

@author: filip

Chapter 9 - Unsupervised Learning Technique

"""

# K-Means

from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=100, centers = 5, n_features = 2, random_state = 0)

from sklearn.cluster import KMeans
import numpy as np

k = 5

kmeans= KMeans (n_clusters = k)
y_pred = kmeans.fit_predict(X)

kmeans.cluster_centers_

X_new = np.array([[0,2],[3,2],[-3,3],[-3,2.5]])

kmeans.predict(X_new)

#Distances from each instance to every centroid

kmeans.transform(X_new)

#Choose the best initial centroids

good_init = np.array([[-3,3],[-3,2],[-3,1],[-1,2],[0,2]])
kmeans = KMeans(n_clusters=5, init = good_init, n_init = 1)

from sklearn.cluster import MiniBatchKMeans

minibatch_kmeans = MiniBatchKMeans(n_clusters = 5)
minibatch_kmeans.fit(X)

#Silhouette Score

from sklearn.metrics import silhouette_score
silhouette_score(X, kmeans.labels_)

#Using clustering for Image Segmentation

from matplotlib.image import imread
import os 

image = imread(os.path.join("images","unsupervisedlearning","ladybug.png"))

X = image.reshape(-1,3)

kmeans = KMeans(n_clusters=8).fit(X)

segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(image.shape)

#Using Clustering for Preprocessing

from sklearn.datasets import load_digits

X_digits, y_digits = load_digits(return_X_y=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

log_reg.score(X_test,y_test)

#Increase accuracy by KMeans++

from sklearn.pipeline import Pipeline

pipeline = Pipeline([("kmeans",KMeans(n_clusters=50)),("log_reg",LogisticRegression())])

pipeline.fit(X_train, y_train)

pipeline.score(X_test, y_test)

from sklearn.model_selection import GridSearchCV

param_grid = dict(kmeans__n_clusters=range(2,100))
grid_clf = GridSearchCV(pipeline,param_grid, cv=3, verbose=2)
grid_clf.fit(X_train, y_train)

grid_clf.best_params_
grid_clf.score(X_test,y_test)

#Using Clustering for Semi-Supervised Learning

n_labeled = 50
log_reg = LogisticRegression()
log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])

log_reg.score(X_test, y_test)

k = 50
kmeans = KMeans(n_clusters=k)
X_digits_dist = kmeans.fit_transform(X_train)
representative_digit_idx = np.argmin(X_digits_dist, axis=0)
X_representative_digits = X_train[representative_digit_idx]

y_representative_digits = np.array([4,8,0,6,8,3,7,7,9,1,5,5,8,5,2,1,2,9,6,1,1,6,9,0,8,3,8,7,4,1,6,5,2,4,1,8,6,3,9,2,4,2,9,4,7,6,2,3,1,1])

log_reg = LogisticRegression()
log_reg.fit(X_representative_digits, y_representative_digits)
log_reg.score(X_test, y_test)

#DBSCAN

from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

X,y = make_moons(n_samples = 1000, noise = 0.05)
dbscan = DBSCAN(eps = 0.05, min_samples = 5)
dbscan.fit(X)

dbscan.labels_

len(dbscan.core_sample_indices_)

dbscan.core_sample_indices_
dbscan.components_

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])

X_new = np.array([[-0.5,0],[0,0.5],[1,-0.1],[2,1]])
knn.predict(X_new)
knn.predict_proba(X_new)

#Gaussian Mixtures

from sklearn.mixture import GaussianMixture

gm = GaussianMixture(n_components = 3, n_init=10)
gm.fit(X)

gm.weights_
gm.means_
gm.covariances_

gm.predict(X)
gm.predict_proba(X)
ente