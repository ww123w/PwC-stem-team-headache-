import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir("../../Data")

#Data
Revenue = pd.read_csv('AAHK_revenue.csv')
Revenue_ = []
Passenger = pd.read_csv('passenger_yearly.csv')
Passenger_ = []

#Correlation
for i in range(Revenue['revenue'].size):
    Revenue_.append(Revenue['revenue'][i])
    Passenger_.append(Passenger['total'][i])

print(np.corrcoef(Revenue_,Passenger_))

#Visualize the data
fig, ax1 = plt.subplots()

color = 'tab:red'
plt.title('Relationship Between Revenue of AAHK and No. of Passengers')
ax1.set_ylabel('Passenger Traffic(millions of passengers)', color=color)
ax1.plot(Passenger['total'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_xticklabels([1998,2000,2003,2006,2009,2011,2013,2015,2017,2020])
ax2.set_ylabel('Revenue(in HK$ million)', color=color)
ax2.plot(Revenue['revenue'], color=color)
ax2.tick_params(axis='y', labelcolor=color)


fig.tight_layout()
plt.show()

