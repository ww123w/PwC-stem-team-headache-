# Calculating the correlation between cases of Covid19 and aircraft travel volumn in Hong Kong
# Coefficient c lies between -1 and 1, c = 0 means no correlation and |c| = 1 means positive/negative correlated.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

#data
Stock_price = yf.download('0293.HK', '2000-1-04', '2020-07-03',interval = "1mo")

#using local ### to be correct
Passenger = pd.read_csv('/Users/alanngo/PycharmProjects/PwC-stem-team-headache-/Data/HK_aircraft_til2020Jun_cleaned2.csv')
Passenger['Total_'].index = pd.date_range(start='2000-1-01', end = '2020-07-01', freq='M')

print(Stock_price['Adj Close'])
print(Passenger['Total_'])