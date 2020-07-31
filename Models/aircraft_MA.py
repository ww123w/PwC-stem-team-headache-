import pandas as pd
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

os.chdir("../Data")
aircraft = pd.read_csv("aircraft_2000_2020.csv")
aircraft['Total_'].index = pd.date_range(start='2000-1-01', end='2020-07-01', freq='M')

#plt.plot(aircraft['Total_'])
#plt.show()

# 2003 SARS & 2008 Financial crisis -> noise. Therefore, We take the data from 2010-01-01 (row 120) to 2018-12-31 (row 228) (108 months)
data_input = aircraft['Total_'][:]
print(data_input)

# Rolling mean & Stationarity test
def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = timeseries.rolling(12).mean() # 12 means 12 months/1 year
    rolstd = timeseries.rolling(12).std()

    # Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Total no. of Passengers Arrival and Departure from Hong Kong')
    plt.ylabel('Total no. of passengers')
    plt.xlabel('From Jan 2000 to Jun 2020')

    plt.show()

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

test_stationarity(data_input)
