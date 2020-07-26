import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import os
#data
Stock_price = yf.download('0293.HK', '2000-01-04', '2020-07-03')
Stock_price_ =[]
print(Stock_price)

os.chdir("../../Data")
Passenger = pd.read_csv('HK_aircraft_til2020Jun_cleaned2.csv')
Passenger['Total_'].index = pd.date_range(start='2000-1-01', end='2020-07-01', freq='M')
print(Passenger['Total_'])
Passenger_ = []

#Correlation
for i in range(Passenger['Total_'].size):
    Passenger_.append(Passenger['Total_'][i])
    Stock_price_.append(Stock_price['Adj Close'][i])

print(np.corrcoef(Stock_price_,Passenger_))


#Visualize the data
fig, ax1 = plt.subplots()

color = 'tab:red'
plt.title('Relationship Between Stock Price and Total No. of Passengers')
ax1.set_xlabel('Time Period')
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