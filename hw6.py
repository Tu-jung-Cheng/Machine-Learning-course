#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 19:14:22 2019

@author: shelley
"""

import robert.preprocessor as pp

# Load Dataset
dataset = pp.dataset(file="Device_Failure.csv")

# Decomposition of Variables
X, Y = pp.decomposition(dataset, x_columns=[0], y_columns=[1])

# Training / Testing Set
X_train, X_test, Y_train, Y_test = pp.split_train_test(x_ary=X, y_ary=Y, train_size=0.8)

# In[] Linear Regression as comparison
from robert.regression import SimpleRegressor
import robert.model_drawer as md

reg_simple = SimpleRegressor()
Y_simple = reg_simple.fit(x_train=X, y_train=Y).predict(x_test=X)

md.sample_model(sample_data=(X, Y), model_data=(X, Y_simple))
print("R-Squared of Simple Regression:", reg_simple.r_score(x_test=X, y_test=Y))


# In[] Polynomial Regression with Robert's Class
from robert.regression import PolynomialRegressor

reg_poly = PolynomialRegressor()
reg_poly.best_degree(x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test, verbose=True)

Y_poly = reg_poly.fit(x_train=X, y_train=Y).predict(x_test=X)
md.sample_model(sample_data=(X, Y), model_data=(X, Y_poly))


# In[] 機器使用年限
import pandas as pd

work_years = float(input("請輸入使用年份："))

pred = reg_poly.predict(pd.DataFrame([[work_years]])).iloc[0, 0]
avg = pred/work_years
print("總失效時間 {:.2f}小時 ".format(pred))
print("每年平均失效時間 {:.2f}小時/年".format(avg))


# In[] Comparing the two models
from robert.performance import rmse

rmse_linear = rmse(Y, Y_simple)
rmse_poly = rmse(Y, Y_poly)

best_deg = reg_poly.best_degree(x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test, verbose=False)
print("RMSE :{:.4f},best degree:{:}".format(rmse_poly,best_deg))

# In[] Check for Assumption of Linear Regression

#from robert.criteria import AssumptionChecker

#checker = AssumptionChecker(X, X, Y, Y, Y_poly)
#checker.check_all()

