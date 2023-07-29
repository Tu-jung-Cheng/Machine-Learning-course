#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 18:38:42 2019

@author: shelley
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 18:15:55 2019

@author: shelley
"""

import robert.preprocessor as pp

# Load Data
dataset = pp.dataset(file="Diabetes.csv")

# In[]
# X, Y decomposition
X, Y = pp.decomposition(dataset, x_columns=[0,1, 2, 3,4,5,6,7], y_columns=[8])

# One-Hot Encoder
#X = pp.onehot_encoder(ary=X, columns=[0],remove_trap = True)


# Feature Selection
from robert.preprocessor import KBestSelector
selector = KBestSelector()
X = selector.fit(x_ary=X, y_ary=Y, auto=True, verbose=True, sort=True).transform(x_ary=X)

# Split Training / TEsting Set
X_train, X_test, Y_train, Y_test = pp.split_train_test(x_ary=X, y_ary=Y)

# Feature Scaling
X_train, X_test = pp.feature_scaling(fit_ary=X_train, transform_arys=(X_train, X_test))

# In[] Training & Testing using Robert's Class
from robert.classification import NaiveBayesClassifier

classifier = NaiveBayesClassifier()
Y_pred = classifier.fit(X_train, Y_train).predict(X_test)

# In[] Performance
from robert.performance import ClassificationPerformance

pfm = ClassificationPerformance(Y_test, Y_pred)
print("Confusion Matrix:\n", pfm.confusion_matrix())
print("Accuracy: {:.2%}".format(pfm.accuracy()))
print("Recall: {:.2%}".format(pfm.recall()))
print("Precision: {:.2%}".format(pfm.precision()))
print("F1-score: {:.2%}".format(pfm.f_score()))


from robert.performance import KFoldClassificationPerformance

K = 10
kfp = KFoldClassificationPerformance(x_ary=X, y_ary=Y, classifier=classifier.classifier, k_fold=K, verbose=False )

print("{} Folds Mean Accuracy: {}".format(K, kfp.accuracy()))
print("{} Folds Mean Recall: {}".format(K, kfp.recall()))
print("{} Folds Mean Precision: {}".format(K, kfp.precision()))
print("{} Folds Mean F1-Score: {}".format(K, kfp.f_score()))


# In[] Visualization
import robert.model_drawer as md
selector = KBestSelector(best_k = 2)
X = selector.fit(x_ary=X, y_ary=Y).transform(x_ary=X)

# Split Training / TEsting Set
X_train, X_test, Y_train, Y_test = pp.split_train_test(x_ary=X, y_ary=Y)

# Feature Scaling
X_train, X_test = pp.feature_scaling(fit_ary=X_train, transform_arys=(X_train, X_test))

# In[] Training & Testing using Robert's Class
from robert.classification import NaiveBayesClassifier

classifier = NaiveBayesClassifier()
Y_pred = classifier.fit(X_train, Y_train).predict(X_test)

md.classify_result(x=X_train, y=Y_train, classifier=classifier, title="訓練集 vs. 模型")
md.classify_result(x=X_test, y=Y_test, classifier=classifier, title="測試集 vs. 模型")


# In[] Check for Variables Independence
from robert.criteria import AssumptionChecker

checker = AssumptionChecker(x_train=X_train, x_test=X_test, y_train=Y_train, y_test=Y_test, y_pred=Y_pred)
checker.features_correlation(heatmap=True)



## 加分題