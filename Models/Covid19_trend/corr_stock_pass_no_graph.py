# Calculating the correlation between cases of Covid19 and aircraft travel volumn in Hong Kong
# Coefficient c lies between -1 and 1, c = 0 means no correlation and |c| = 1 means positive/negative correlated.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import os

#data

Stock_price = yf.download('0293.HK', '2000-2-01', '2020-07-03',interval = "1mo")
print(Stock_price)
Stock = []

#using local ### to be correct
os.chdir("../../Data")
Passenger = pd.read_csv('aircraft_2000_2020.csv')
Passenger['Total_'].index = pd.date_range(start='2000-01-01', end = '2020-07-01', freq='M')
print(Passenger['Total_'])
Passenger_list = []

for i in range((Stock_price['Adj Close'].size)):
    if not np.isnan(Stock_price['Adj Close'][i]):
        Stock.append(Stock_price['Adj Close'][i])

for i in range(Passenger['Total_'].size):
    Passenger_list.append(Passenger['Total_'][i])

print(np.corrcoef(Stock,Passenger_list))