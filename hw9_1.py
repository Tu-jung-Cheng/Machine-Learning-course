#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 19:11:04 2019

@author: shelley
"""

import robert.preprocessor as pp

# Load Dataset
dataset = pp.dataset(file="Voice.csv")

# In[]
# Decomposition of Variables
X, Y = pp.decomposition(dataset, x_columns=[i for i in range(19)], y_columns=[20])

# Apply One Hot Encoder 
Y = pp.onehot_encoder(Y, columns=[0],remove_trap = True)

# Feature Selection
from robert.preprocessor import KBestSelector
#selector = KBestSelector(best_k=2)
selector = KBestSelector()

X = selector.fit(x_ary=X, y_ary=Y, auto=True, verbose=True, sort=True).transform(x_ary=X)

# Split Training / TEsting Set
X_train, X_test, Y_train, Y_test = pp.split_train_test(x_ary=X, y_ary=Y)

# Feature Scaling
X_train, X_test = pp.feature_scaling(fit_ary=X_train, transform_arys=(X_train, X_test))

# In[]

from sklearn.svm import SVC
import time

classifier = SVC(C=1.0, kernel="rbf", gamma="scale", random_state=int(time.time()))
classifier.fit(X_train, Y_train.values.ravel())
Y_pred = classifier.predict(X_test)


# In[] SVM with Robert's SVM class
from robert.classification import SVM

classifier = SVM()


# Train & Predict
#classifier = SVM(C=0.01, kernel="linear", coef0=1.0)
Y_pred_svm = classifier.fit(X_train, Y_train).predict(X_test)
#
# In[] Performance (Note: Never try kernel="linear" of SVC() for K-fold, it will hang!!)
from robert.performance import KFoldClassificationPerformance

K = 10
kfp = KFoldClassificationPerformance(x_ary=X, y_ary=Y, classifier=classifier.classifier, k_fold=K)

print("----- SVM Classification -----")
print("{} Folds Mean Accuracy: {}".format(K, kfp.accuracy()))
print("{} Folds Mean Recall: {}".format(K, kfp.recall()))
print("{} Folds Mean Precision: {}".format(K, kfp.precision()))
print("{} Folds Mean F1-Score: {}".format(K, kfp.f_score()))

