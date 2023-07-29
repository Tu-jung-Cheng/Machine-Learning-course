#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 18:54:57 2019

@author: shelley
"""
import matplotlib.pyplot as plt
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader as data
import pandas as pd
from pandas_datareader._utils import RemoteDataError

# In[] word type
plt.rcParams['font.sans-serif']=['Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus']=False


# In[]
stockID = pd.read_csv("TaiwanStockID.csv")


stockName = input("請輸入股票名稱或代號:")

if stockName.isdigit() == False:
    stockNN = stockName
    condition = stockID.StockName  ==stockName
    stockName = stockID[condition].iloc[0]["StockID"]



# In[]
import dateutil.parser as psr 
startDate = psr.parse(input("請輸入查詢起始日期:"))
print(startDate.date())

endDate = psr.parse(input("請輸入查詢終止日期:"))
print(endDate.date())

# In[]

stockQuery = "{}.TW". format(stockName)
try:
    data = data.DataReader(stockQuery, "yahoo", startDate.date(), endDate.date())
    close_price= data["Close"]
    
except ValueError:
    print("start date must be earier than end date")
except RemoteDataError:
    print("Invalid Stock ID:{}".format(stockName))

# In[]
plt.title("{} {}~{} 收盤價".format(stockName,startDate.date(), endDate.date()))
plt.xlabel("date")
plt.ylabel("指數")
    
close_price.plot(label="收盤價")
close_price.rolling(window=20).mean().plot(label="20MA")
close_price.rolling(window=60).mean().plot(label="60MA")
plt.legend(loc = 'best')
plt.show()