#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 21:02:56 2019

@author: shelley
"""

import numpy as np
import pandas as pd

# In[] Read Dataset
dataset = pd.read_csv("HealthCheck.csv")

# In[] Dep & Indep variable
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values

# In[] Missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# In[] Digitalize
from sklearn.preprocessing import LabelEncoder
labelEncoder=LabelEncoder()
Y = labelEncoder.fit_transform(Y).astype("float64")

# Indep variable part
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

dummyEncoder = ColumnTransformer(
        [("City_Encoder", OneHotEncoder(), [0])], 
        remainder="passthrough"
        )

X = dummyEncoder.fit_transform(X).astype("float64")

# In[] train_test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# In[] StandardScaler
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler().fit(X_train)
X_train = sc_X.transform(X_train)
X_test = sc_X.transform(X_test)


# In[] print
print("x_train",X_train)
print("x_test",X_test)
print("y_train",Y_train)
print("y_test",Y_test)




