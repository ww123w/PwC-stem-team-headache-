import pandas as pd
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

os.chdir("../Data")
aircraft = pd.read_csv("aircraft_2000_2020_predict_pess.csv")
aircraft['Total_'].index = pd.date_range(start='2000-1-01', end='2021-09-01', freq='M')

#fig, aircraft_plot = plt.subplots()
plt.title('From Jan 00 to Aug 21' + '\n' + 'Total no. of Passengers Arrival and Departure from Hong Kong' + '\n')
#aircraft_plot.plot((aircraft['Total_'][:-5]), color = 'tab:blue')

#predict = aircraft_plot.twinx()
#predict.plot((aircraft['Total_'][-6:]), color = 'tab:red')
#fig.tight_layout()
plt.plot(aircraft['Total_'])
plt.show()
