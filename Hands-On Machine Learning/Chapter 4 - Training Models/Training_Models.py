# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 13:53:59 2021

@author: filip

Chapter 4 - Training Models

"""

#MSE cost function for a Linear Regression model
# MSE(x,h) = 1/m * Sigma(Theta*x-y)²

#Solution Normal Equation - Closed-form solution - non interactive

#Theta = (X.T*X)^-1*X.T*y 

import numpy as np

np.random.seed(42)

X = 2*np.random.rand(100,1)
y = 4 + 3*X + np.random.rand(100,1)

import matplotlib.pyplot as plt

plt.scatter(X,y)
plt.ylim(0,14)
plt.show()

#Solving the Normal Equation
X_b = np.c_[np.ones((100,1)),X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

#Predicting with Theta
X_new = np.array([[0],[2]])
X_new_b = np.c_[np.ones((2,1)),X_new] # add x0 = 1 to each iinstance
y_predict = X_new_b.dot(theta_best)

plt.plot(X_new, y_predict, "r-")
plt.plot(X,y, "b.")
plt.axis([0,2,0,15])

#Linear Regression using Scikit-Learn
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)
lin_reg.intercept_,lin_reg.coef_

lin_reg.predict(X_new)

#Linear Regression is based on Least Squares
theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b,y,rcond=1e-6)

#Gradient Descent
#d(MSE)/dtheta = 2/m *sum(theta.T*x-y)*x
# theta(k+1) = theta(k) - alpha*Nabla(theta)MSE(thehta)
 
alpha = 0.1 
n_interations = 1000
m = 100

theta = np.random.rand(2,1) 

for i in range(n_interations):
    gradients = (2/m)*X_b.T.dot(X_b.dot(theta)-y)
    theta = theta - alpha*gradients
    
#SGD Regressor
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=2000, tol=1e-3, penalty=None, eta0=0.1)
sgd_reg.fit(X,y.ravel())
sgd_reg.intercept_, sgd_reg.coef_

#Polynomial Regression
m = 100
X = 6*np.random.rand(m,1)-3
X = np.sort(X,axis=0)
y = 0.5*X**2 + X+2 + np.random.rand(m,1)

plt.scatter(X,y)
plt.ylim(0,10)

from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
X[0]
X_poly[0]

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_,lin_reg.coef_

Yhat = lin_reg.predict(X_poly)

    
plt.scatter(X,y)
plt.plot(X,Yhat,color='red')
plt.ylim(0,10)

poly_features1 = PolynomialFeatures(degree=1, include_bias=False)
X1_poly = poly_features1.fit_transform(X)
lin_reg1 = LinearRegression()
lin_reg1.fit(X1_poly, y)
Yhat1 = lin_reg1.predict(X1_poly)

plt.scatter(X,y)
plt.plot(X,Yhat,color='red')
plt.plot(X,Yhat1,color='blue')
plt.ylim(0,10)

poly_features2 = PolynomialFeatures(degree=50, include_bias=False)
X2_poly = poly_features2.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X2_poly, y)
Yhat2 = lin_reg2.predict(X2_poly)

plt.scatter(X,y)
plt.plot(X,Yhat,color='red')
plt.plot(X,Yhat1,color='blue')
plt.plot(X,Yhat2,color='green')
plt.ylim(0,10)

#learning curves

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)
    train_errors, val_errors = [], []
    for m in range(1,len(X_train)):
        model.fit(X_train[:m],y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m],y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    plt.plot(np.sqrt(train_errors),"r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors),"b-", linewidth=3, label="val")
    plt.ylim(0,3)
    plt.legend(["train","val"])
    plt.xlabel("Training set size")
    plt.ylabel("RMSE")
    
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X,y)

from sklearn.pipeline import Pipeline
polynomial_regression = Pipeline([
    ("poly_features", PolynomialFeatures(degree=5, include_bias=False)),
    ("lin_reg",LinearRegression()),])

plot_learning_curves(polynomial_regression, X,y)

#Regularization
#Closed form: André-Louis Cholesky

from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=.1, solver="cholesky")
ridge_reg.fit(X,y)
ridge_reg.predict([[1.5]])

sgd_reg = SGDRegressor(penalty="l2")
sgd_reg.fit(X,y.ravel())
sgd_reg.predict([[1.5]])

plt.scatter(X,y)
plt.scatter(ridge_reg.predict(X),y)
plt.scatter(sgd_reg.predict(X),y)

#Lasso Regression
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X,y)
lasso_reg.predict([[1.5]])

#Elastic Net
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=.1,l1_ratio=0.5)
elastic_net.fit(X,y)
elastic_net.predict([[1.5]])

#Early Stopping
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)

poly_scaler = Pipeline([
    ("poly_features",PolynomialFeatures(degree=90, include_bias=False)),
    ("std_scaler", StandardScaler())
    ])

X_train_poly_scaled = poly_scaler.fit_transform(X_train)
X_val_poly_scaled = poly_scaler.transform(X_val)

sgd_reg = SGDRegressor(max_iter=1, tol= -np.infty, warm_start=True,
                       penalty=None, learning_rate="constant",eta0=0.0005)
minimum_val_error = float("inf")
best_epoch = None
best_model = None
for epoch in range(1000):
    sgd_reg.fit(X_train_poly_scaled, y_train)
    y_val_predict = sgd_reg.predict(X_val_poly_scaled)
    val_error = mean_squared_error(y_val, y_val_predict)
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = clone(sgd_reg)
        
        
#Logistic regression
from sklearn import datasets

iris = datasets.load_iris()
list(iris.keys())

X = iris["data"][:,3:] #petal width
y = (iris["target"] == 2).astype(np.int) #select iris virginica

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X,y)

X_new = np.linspace(0,3,1000).reshape(-1,1)
y_proba = log_reg.predict_proba(X_new)
plt.plot(X_new, y_proba[:,1], "g-", label="Iris virginica")
plt.plot(X_new, y_proba[:,0], "b--", label="Not iris virginica")
plt.legend()

plt.scatter(iris["data"][:,2],iris["data"][:,3])
plt.xlabel("Petal length")
plt.ylabel("Petal width")
