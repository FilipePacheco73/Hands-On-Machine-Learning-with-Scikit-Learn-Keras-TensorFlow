# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 09:22:28 2021

@author: Filipe Pacheco

Hands-On Machine Learning

Chapter 2 - End to End Machine Learning Project 

MPL to learn the price of house in a region

"""

import os 
import tarfile
import urllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from pandas.plotting import scatter_matrix

# Original path to download the dataset

"""
DOWNLOAD_PATH = "https://github.com/ageron/handson-ml2/tree/master/"
HOUSING_PATH = os.path.join("datasets","housing")
HOUSING_URL = DOWNLOAD_PATH + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url = HOUSING_URL, housing_path = HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url,tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    
fetch_housing_data()

"""

#initials insights about the data

housing = pd.read_csv("housing.csv")
housing.info()
housing["ocean_proximity"].value_counts()
housing.describe()
housing.hist(bins = 50, figsize = (20,15))
plt.show

#division of datas from test and train

def split_train_test(data,test_ratio):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)

train_set, test_set = train_test_split(housing, test_size = .2, random_state = 42)

#criation of strata, sub-representation of the total

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins = [0,1.5,3,4.5,6, np.inf],
                               labels = [1,2,3,4,5])

housing["income_cat"].hist()

split = StratifiedShuffleSplit(n_splits = 1, test_size = .2, random_state = 42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    
strat_test_set["income_cat"].value_counts()/len(strat_test_set)

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis = 1, inplace = True)

#Print some good features
    
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=.1)
# alpha make easier to see the high density points
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=.4,
             s= housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap= plt.get_cmap("jet"),colorbar=True)
plt.legend()
    

#Looking for correlations - Person's

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,8))

housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)

#data cleasing

housing = strat_train_set.drop("median_house_value",axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

#Treating missing fields - possibilies
 
#option 1
# housing.dropna(subset=["total_bedrooms"]) 

#options 2
# housing.drop("total_bedrooms",axis1) 

#option 3
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median,inplace=True)

#option 4
# imputer = SimpleImputer(missing_values=np.nan,strategy="mean") #option 4
# imputer.fit(housing.iloc[:,:8])
# housing_treated = (imputer.transform(housing.iloc[:,:8]))

# creating output data as integer
ordinal_enconder = OrdinalEncoder()
housing_cat_encoded = ordinal_enconder.fit_transform(housing[["ocean_proximity"]])

# split train and test data for validation

X = np.array(housing.iloc[:,:8]) # data as input
y = housing_cat_encoded # data as output

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42) 

#Linear Regression - train and test

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Evaluation of the training
r2score = r2_score(y_test, lin_reg.predict(X_test)) 
print(r2score)