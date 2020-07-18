import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#data
Stock_price = pd.read_csv('0293.HK copy.csv')
Passenger = pd.read_csv('HK_aircraft_til2020Jun_cleaned2.csv')

#Visualize the data
plt.figure(figsize=(12.5,4.5))
plt.plot(Stock_price['Adj Close'], label='CX')
plt.plot(Passenger['Total_'], label='Total Passenger')
plt.title('Relationship Between Stock Price and Total Number of Passengers')
plt.xlabel('Time Period')
plt.ylabel('...')
plt.legend(loc='upper left')
plt.show()