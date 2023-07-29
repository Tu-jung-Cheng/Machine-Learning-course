#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 10:57:27 2019

@author: shelley
"""

import tensorflow
import keras

# In[] Preprocessing
import robert.preprocessor as pp

# Load Data
dataset = pp.dataset(file="Churn_Modelling.csv")


# Decomposition
X, Y = pp.decomposition(dataset, x_columns=[i for i in range(3, 13)], y_columns=[13])
# In[]
# Dummy Variables
X = pp.onehot_encoder(X, columns=[1,2], remove_trap=True)
Y = pp.label_encoder(Y)

# Feature Selection
from robert.preprocessor import KBestSelector
selector = KBestSelector(best_k="auto")
X = selector.fit(x_ary=X, y_ary=Y, verbose=True, sort=True).transform(x_ary=X)

# Split Training / TEsting Set
X_train, X_test, Y_train, Y_test = pp.split_train_test(x_ary=X, y_ary=Y)

# Feature Scaling (optional)
X_train, X_test = pp.feature_scaling(fit_ary=X_train, transform_arys=(X_train, X_test))

# In[] Neural Networks without Robert's Class
from keras.models import Sequential
from keras.layers import Dense

# Initialize the whole Neural Networks
classifier = Sequential()

# Add the Input & First Hidden Layer
classifier.add(Dense(input_dim=X_train.shape[1], units=45, kernel_initializer="random_normal", activation="relu"))

# Add the Second Hidden Layer
classifier.add(Dense(units=23, kernel_initializer="random_normal", activation="relu"))

# Add the Output Layer
classifier.add(Dense(units=1, kernel_initializer="random_normal", activation="sigmoid"))  

# Compile the whole Neural Networks
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Fit
classifier.fit(x=X_train, y=Y_train, batch_size=10, epochs=100)

# Predict
import pandas as pd
Y_pred = classifier.predict(x=X_test)
Y_pred = (Y_pred > 0.5).astype(int)
Y_pred = pd.DataFrame(Y_pred, index=Y_test.index, columns=Y_test.columns)

# In[] Performance
from robert.performance import ClassificationPerformance

pfm = ClassificationPerformance(Y_test, Y_pred)

print("Confusion Matrix:\n", pfm.confusion_matrix())
print("Accuracy: {:.2%}".format(pfm.accuracy()))
print("Recall: {:.2%}".format(pfm.recall()))
print("Precision: {:.2%}".format(pfm.precision()))
print("F1-score: {:.2%}".format(pfm.f_score()))