#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 21:15:12 2019

@author: shelley
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 07:30:11 2019

@author: 俊男
"""

# In[] Preprocessing
import robert.preprocessor as pp

# Load Data
dataset = pp.dataset(file="HR-Employee-Attrition.csv")

# Decomposition
X, Y = pp.decomposition(dataset, x_columns=[i for i in range(35) if i != 1],  y_columns=[1])

# Dummy Variables
X = pp.onehot_encoder(X, columns=[1, 3, 6, 10, 14, 16, 20, 21], remove_trap=True)
Y, Y_mapping = pp.label_encoder(Y, mapping=True)

# Feature Selection
from robert.preprocessor import KBestSelector
selector = KBestSelector()
X = selector.fit(x_ary=X, y_ary=Y, auto = True, verbose=True, sort=True).transform(x_ary=X)

# Split Training / TEsting Set
X_train, X_test, Y_train, Y_test = pp.split_train_test(x_ary=X, y_ary=Y)

# In[] Decision Tree

# With Robert's Class
from robert.classification import DecisionTree

classifier = DecisionTree()
Y_pred = classifier.fit(X_train, Y_train).predict(X_test)

# In[] Performance
from robert.performance import KFoldClassificationPerformance

K = 10
kfp = KFoldClassificationPerformance(x_ary=X, y_ary=Y, classifier=classifier.classifier, k_fold=K)

print("----- Decision Tree Classification -----")
print("{} Folds Mean Accuracy: {}".format(K, kfp.accuracy()))
print("{} Folds Mean Recall: {}".format(K, kfp.recall()))
print("{} Folds Mean Precision: {}".format(K, kfp.precision()))
print("{} Folds Mean F1-Score: {}".format(K, kfp.f_score()))

# In[] Visualization
from sklearn import tree
import pydotplus
from IPython.display import Image
#import os

#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
cls_name = [Y_mapping[key] for key in sorted(Y_mapping.keys())]
dot_data = tree.export_graphviz(classifier.classifier, filled=True, feature_names=X_test.columns, class_names=cls_name, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

# In[]
from graphviz import Source
temp = dot_data 
s = Source(temp, filename="test.gv", format="png")
s.view()



