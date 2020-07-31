import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller

os.chdir("../Data")
aircraft = pd.read_csv("aircraft_2000_2020_predict_opti.csv")
aircraft['Total_'].index = pd.date_range(start='2000-1-01', end='2021-09-01', freq='M')

#fig, aircraft_plot = plt.subplots()
plt.title('Total no. of Passengers Arrival and Departure from Hong Kong')
plt.ylabel('Total no. of passengers')
plt.xlabel('From Jan 2000 to Aug 2021')
#aircraft_plot.plot((aircraft['Total_'][:-5]), color = 'tab:blue')

#predict = aircraft_plot.twinx()
#predict.plot((aircraft['Total_'][-6:]), color = 'tab:red')
#fig.tight_layout()
plt.plot(aircraft['Total_'])
plt.show()

#graph 2
aircraft = pd.read_csv("aircraft_2019_predict_opti.csv")
aircraft['Total_'].index = pd.date_range(start='2019-1-01', end='2021-09-01', freq='M')
fig, ax1 = plt.subplots()
color = 'tab:blue'
plt.title('Total no. of Passengers Arrival and Departure from Hong Kong')
ax1.plot(aircraft['Total_'], color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticklabels(['Jan','May','Sep','Jan','May','Sep','Jan','May','Sep'])
plt.ylabel('Total no. of passengers')
plt.xlabel('From Jan 2019 to Aug 2021')
plt.plot(aircraft['Total_'])
plt.show()