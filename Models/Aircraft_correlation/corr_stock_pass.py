import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import os
#data
Stock_price = yf.download('0293.HK', '2000-01-31', '2020-07-03', interval="1mo")
Stock_price_ =[]

os.chdir("../../Data")

Passenger = pd.read_csv('aircraft_2000_2020.csv')
Passenger['Total_'].index = pd.date_range(start='2000-1-01', end='2020-07-01', freq='M')
Passenger_ = []

#Correlation
for i in range(Passenger['Total_'].size):
    Passenger_.append(Passenger['Total_'][i])

for i in range((Stock_price['Adj Close'].size)):
    if not np.isnan(Stock_price['Adj Close'][i]):
        Stock_price_.append(Stock_price['Adj Close'][i])

print(np.corrcoef(Stock_price_,Passenger_))


#Visualize the data
fig, ax1 = plt.subplots()

Stock_price = yf.download('0293.HK', '2000-01-04', '2020-07-03')

color = 'tab:red'
plt.title('Relationship Between No. of Passengers and CX Stock Price')
ax1.set_ylabel('Adj. Close Price HKD ($)', color=color)
ax1.plot(Stock_price['Adj Close'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('Total Number of Passengers', color=color)
ax2.plot(Passenger['Total_'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()