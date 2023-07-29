#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 20:22:49 2019

@author: shelley
"""

# In[] Preprocessing
import robert.preprocessor as pp

# Load Dataset
dataset = pp.dataset(file="CreditCards.csv")


# Decomposition
X = pp.decomposition(dataset, x_columns=[i for i in range(1,18)])

# Missing data
X = pp.missing_data(X, strategy="mean")
# Feature Scaling (for PCA Feature Selection)
X = pp.feature_scaling(fit_ary=X, transform_arys=X)

# In[]
# Feature Selection (PCA)
from robert.preprocessor import PCASelector

selector = PCASelector(best_k=2)
X = selector.fit(x_ary=X, verbose=True, plot=True).transform(x_ary=X)



# In[] K-Means Clustering (With Robert's Class)
from robert.clustering import KMeansCluster

cluster = KMeansCluster()
Y_pred = cluster.fit(x_ary=X, verbose=True, plot=True).predict(x_ary=X, y_column="Customer Type")

# Optional, Attach the Y_pred to Dataset & Save as .CSV file
dataset = pp.combine(dataset, Y_pred)
dataset.to_csv("CreditCards_answers.csv")


# In[] Visualization (With Robert's Class)
import robert.model_drawer as md

md.cluster_drawer(x=X, y=Y_pred, centroids=cluster.centroids, title="Customers Segmentation")

# In[] k_best
print('----------------k best-------------------')
# In[] Preprocessing
import robert.preprocessor as pp

# Load Data, also can be loaded by sklearn.datasets.load_wine()
dataset = pp.dataset(file="CreditCards_answers.csv")

# In[]
# Decomposition
X, Y = pp.decomposition(dataset, x_columns=[i for i in range(2,19)], y_columns=[19])
X = pp.missing_data(X, strategy="mean")

# In[]
# By KBestSelector
from robert.preprocessor import KBestSelector
selector = KBestSelector()
X = selector.fit(x_ary=X, y_ary=Y, verbose=True, sort=True).transform(x_ary=X)
# Split Training / TEsting Set
X_train, X_test, Y_train, Y_test = pp.split_train_test(x_ary=X, y_ary=Y)

# Feature Scaling
X_train, X_test = pp.feature_scaling(fit_ary=X_train, transform_arys=(X_train, X_test))

# In[] Random Forest
# With Robert's Class   
from robert.classification import RandomForest

classifier = RandomForest(n_estimators=10, criterion="entropy")
Y_pred = classifier.fit(X_train, Y_train).predict(X_test)

# In[] Performance
from robert.performance import KFoldClassificationPerformance

K = 10
kfp = KFoldClassificationPerformance(x_ary=X, y_ary=Y, classifier=classifier.classifier, k_fold=K)

print("----- Random Forest Classification -----")
print("{} Folds Mean Accuracy: {}".format(K, kfp.accuracy()))
print("{} Folds Mean Recall: {}".format(K, kfp.recall()))
print("{} Folds Mean Precision: {}".format(K, kfp.precision()))
print("{} Folds Mean F1-Score: {}".format(K, kfp.f_score()))

# In[] Visualization
import robert.model_drawer as md
from IPython.display import Image

clfr = classifier.classifier.estimators_[0]
graph = md.tree_drawer(classifier=clfr, feature_names=X_test.columns, graphviz_bin='C:/Program Files (x86)/Graphviz2.38/bin/')
Image(graph.create_png())

