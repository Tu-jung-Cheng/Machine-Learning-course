#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 20:25:29 2019

@author: shelley
"""

# In[1]
import numpy as np

dice = ["1","2","3","4","5","6"]
reality  = np.random.choice(dice, size = (100),replace = True, p = [0.1,0.1,0.2,0.1,0.2,0.3])
print("100 times roll the unfair dice:")
print(reality)

print("theoretical probability:",dict(zip(dice,[0.1,0.1,0.2,0.1,0.2,0.3])))
u, indices = np.unique(reality,return_counts=True)

print("real probability:",dict(zip(u, indices/100)))