#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 18:46:55 2019

@author: shelley
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:57:58 2019

@author: 俊男
"""
print('----------------PCA-------------------')

# In[] Preprocessing
import robert.preprocessor as pp

# Load Data, also can be loaded by sklearn.datasets.load_wine()
dataset = pp.dataset(file="Zoo_Data.csv")
dataset_classname = pp.dataset("Zoo_Class_Name.csv")
class_names = [row['Class_Type'] for index, row in dataset_classname.iterrows()]
#print(class_names)
target_names=['Mammal', 'Bird', 'Reptile', 'Fish', 'Amphibian', 'Bug', 'Invertebrate']
# Decomposition
X, Y = pp.decomposition(dataset, x_columns=[i for i in range(1,17)], y_columns=[17])
# ----- Feature Selection by PCA (PCA requires Feature Scaling) -----
X = pp.feature_scaling(fit_ary=X, transform_arys=X)

# In[]

# With Robert's Class
from robert.preprocessor import PCASelector

selector = PCASelector(best_k="auto")
X = selector.fit(x_ary=X, verbose=True, plot=True).transform(X)
#
## Split Training / TEsting Set
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
graph = md.tree_drawer(classifier=clfr, feature_names=X_test.columns, target_names=target_names, graphviz_bin='C:/Program Files (x86)/Graphviz2.38/bin/')
Image(graph.create_png())

# In[] k_best
print('----------------k best-------------------')
# In[] Preprocessing
import robert.preprocessor as pp

# Load Data, also can be loaded by sklearn.datasets.load_wine()
dataset = pp.dataset(file="Zoo_Data.csv")
dataset_classname = pp.dataset("Zoo_Class_Name.csv")
class_names = [row['Class_Type'] for index, row in dataset_classname.iterrows()]
#print(class_names)
target_names=['Mammal', 'Bird', 'Reptile', 'Fish', 'Amphibian', 'Bug', 'Invertebrate']
# Decomposition
X, Y = pp.decomposition(dataset, x_columns=[i for i in range(1,17)], y_columns=[17])

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
graph = md.tree_drawer(classifier=clfr, feature_names=X_test.columns, target_names=target_names, graphviz_bin='C:/Program Files (x86)/Graphviz2.38/bin/')
Image(graph.create_png())

