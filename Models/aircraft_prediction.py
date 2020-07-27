import pandas as pd
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import numpy as np


os.chdir("../Data")
aircraft = pd.read_csv("aircraft_2000_2020.csv")
aircraft['Total_'].index = pd.date_range(start='2000-1-01', end='2020-07-01', freq='M')

# 2003 SARS & 2008 Financial crisis -> noise. Therefore, We take the data from 2010-01-01 (row 120) to 2018-12-31 (row 228) (108 months)
data_input = aircraft['Total_'][120:228]

# ploting sqrt
data_sqrt = np.sqrt(data_input)
plt.plot(data_sqrt, color ='blue')

# ploting MA
MA = data_sqrt.rolling(12).mean()
plt.plot(MA)
plt.plot(MA, color='red')

plt.show()

sqrt_avg_diff = data_sqrt - MA
sqrt_avg_diff.dropna(inplace=True)

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
    plt.title('From Jun 10 to Dec 18' + '\n'
              'Rolling Mean & Standard Deviation')

    plt.show()

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

test_stationarity(sqrt_avg_diff)

