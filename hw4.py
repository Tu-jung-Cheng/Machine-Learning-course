#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 16:51:17 2019

@author: shelley
"""
# In[]
user_gender = eval(input("請輸入您的性別(1.男 2.女):")) - 1
user_age = eval(input("請輸入您的年齡(6-15):"))
user_height = eval(input("請輸入您的身高(cm):"))
user_weight = eval(input("請輸入您的體重(kg):"))


# In[]
 
from robert import preprocessor as pp

dataset = pp.dataset("Student_Height.csv")
dataset2 = pp.dataset("Student_Weight.csv")
# Independent/Dependent Variables Decomposition
# boy_height
X, Y = pp.decomposition(dataset, [1], [3])
X_train, X_test, Y_train, Y_test = pp.split_train_test(X, Y, train_size=2/3)

# girl_height
X1, Y1 = pp.decomposition(dataset, [1], [4])
X_train1, X_test1, Y_train1, Y_test1 = pp.split_train_test(X1, Y1, train_size=2/3)

# boy_weight
X2, Y2 = pp.decomposition(dataset2, [1], [3])
X_train2, X_test2, Y_train2, Y_test2 = pp.split_train_test(X2, Y2, train_size=2/3)

# girl_weight
X3, Y3 = pp.decomposition(dataset2, [1], [4])
X_train3, X_test3, Y_train3, Y_test3 = pp.split_train_test(X3, Y3, train_size=2/3)


# In[] Fitting Simple Regressor
from robert.regression import SimpleRegressor

regressor00 = SimpleRegressor()
Y_pred00 = regressor00.fit(X_train, Y_train).predict(X_test)

regressor01 = SimpleRegressor()
Y_pred01 = regressor01.fit(X_train1, Y_train1).predict(X_test1)

regressor10 = SimpleRegressor()
Y_pred10 = regressor10.fit(X_train2, Y_train2).predict(X_test2)

regressor11 = SimpleRegressor()
Y_pred11 = regressor11.fit(X_train3, Y_train3).predict(X_test3)


# In[] Visualize the Training Set
from robert import model_drawer as md
if user_gender == 0:
    sample_data=(user_age, user_height)
    model_data=(X_train, regressor00.predict(X_train))
    md.sample_model(sample_data=sample_data, model_data=model_data, title="年齡 vs. 男生身高")
    
    
    sample_data2=(user_age, user_weight)
    model_data2=(X_train2, regressor10.predict(X_train2))
    md.sample_model(sample_data=sample_data2, model_data=model_data2, title="年齡 vs. 男生體重")
    
else :
    sample_data1=(user_age, user_height)
    model_data1=(X_train1, regressor01.predict(X_train1))
    md.sample_model(sample_data=sample_data1, model_data=model_data1, title="年齡 vs. 女生身高")
    
    
    sample_data3=(user_age, user_weight)
    model_data3=(X_train3, regressor11.predict(X_train3))
    md.sample_model(sample_data=sample_data3, model_data=model_data3, title="年齡 vs. 女生體重")

# In[]
if user_gender == 0:

    h_avg = regressor00.predict(x_test=[[user_age]])
    w_avg = regressor10.predict(x_test=[[user_age]])
    print(h_avg,w_avg)
    
else:
    h_avg = regressor01.predict(x_test=[[user_age]])
    w_avg = regressor11.predict(x_test=[[user_age]])
    print(h_avg,w_avg) 

    


