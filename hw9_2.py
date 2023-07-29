#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 19:46:23 2019

@author: shelley
"""

import robert.preprocessor as pp

# Load Dataset
dataset = pp.dataset(file="Voice.csv")

# In[]
# Decomposition of Variables
X, Y = pp.decomposition(dataset, x_columns=[i for i in range(19)], y_columns=[20])


# Apply One Hot Encoder 
Y = pp.onehot_encoder(Y, columns=[0])
# Remove One Column of Categorical Data for avoiding Dummy Variable Trap
Y = pp.remove_columns(Y, [0])

# Feature Selection
from robert.preprocessor import KBestSelector
selector = KBestSelector()
X = selector.fit(x_ary=X, y_ary=Y, auto=True, verbose=True, sort=True).transform(x_ary=X)

# Split Training / TEsting Set
X_train, X_test, Y_train, Y_test = pp.split_train_test(x_ary=X, y_ary=Y)

# Feature Scaling
X_train, X_test = pp.feature_scaling(fit_ary=X_train, transform_arys=(X_train, X_test))


# In[] SVM with Robert's SVM class
from robert.classification import SVM
##Best Parameters: {'C': 1000.0, 'coef0': 1.0, 'kernel': 'linear'}  Best Score: 0.9498646604569702

# Train & Predict
classifier = SVM(C=1000.0, kernel="linear", coef0=1.0)
Y_pred_svm = classifier.fit(X_train, Y_train).predict(X_test)

# In[] Performance (Note: Never try kernel="linear" of SVC() for K-fold, it will hang!!)
from robert.performance import KFoldClassificationPerformance

K = 10
kfp = KFoldClassificationPerformance(x_ary=X, y_ary=Y, classifier=classifier.classifier, k_fold=K)

print("----- SVM Classification -----")
print("{} Folds Mean Accuracy: {}".format(K, kfp.accuracy()))
print("{} Folds Mean Recall: {}".format(K, kfp.recall()))
print("{} Folds Mean Precision: {}".format(K, kfp.precision()))
print("{} Folds Mean F1-Score: {}".format(K, kfp.f_score()))
