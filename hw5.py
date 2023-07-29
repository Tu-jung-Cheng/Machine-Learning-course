#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 02:19:16 2019

@author: shelley
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 12:21:45 2019

@author: 俊男
"""

# In[] Pre-processing
import robert.preprocessor as pp

# Dataset Loading
dataset = pp.dataset("Insurance.csv")

# Independent/Dependent Variables Decomposition
X, Y = pp.decomposition(dataset, [0, 1, 2, 3, 4, 5], [6])


# Apply One Hot Encoder to Column[3]
X = pp.onehot_encoder(X, columns=[1, 4, 5])

# In[]
# Remove One Column of Categorical Data for avoiding Dummy Variable Trap
#X = X[:, 1:]
X = pp.remove_columns(X, [0,2,4])

# Split Training vs. Testing Set
X_train, X_test, Y_train, Y_test = pp.split_train_test(X, Y, train_size=0.8)

# Feature Scaling (optional)
X_train, X_test = pp.feature_scaling(fit_ary=X_train, transform_arys=(X_train, X_test))
Y_train, Y_test = pp.feature_scaling(fit_ary=Y_train, transform_arys=(Y_train, Y_test))

# In[] Create Linear Regressor
from robert.regression import SimpleRegressor

simple_reg = SimpleRegressor()
Y_pred_simple = simple_reg.fit(X_train, Y_train).predict(X_test)

# R-Squared always increase in multiple linear regression --> Use Adjusted R-Squared instead
print("Goodness of Model (R-Squared Score):", simple_reg.r_score(X_test, Y_test))

# In[] Multiple Linear Regression with Robert's class

# Add one constant column
X_train = pp.add_constant(X_train)
X_test = pp.add_constant(X_test)

# In[] Backward Elimination of Features
from robert.regression import MultipleRegressor

regressor = MultipleRegressor()
selected_features = regressor.backward_elimination(x_train=X_train, y_train=Y_train)

Y_predict = regressor.fit(x_train=X_train.iloc[:, selected_features], y_train=Y_train).predict(x_test=X_test.iloc[:, selected_features])

print("Goodness of Model (Adjusted R-Squared Score):", regressor.r_score())

# In[] Check for Assumption of Regression
from robert.criteria import AssumptionChecker

checker = AssumptionChecker(X_train.iloc[:, selected_features], X_test.iloc[:, selected_features], Y_train, Y_test, Y_predict)
checker.y_lim = (-4, 4)
checker.heatmap = True
checker.check_all()
