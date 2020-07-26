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
plt.title('Relationship Between No. of Passengers and Revenue of AAHK')
ax1.set_xlabel('Time Period(Yearly)')
ax1.set_ylabel('Passenger Traffic(millions of passengers)', color=color)
ax1.plot(Passenger['total'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('Revenue(in HK$ million)', color=color)
ax2.plot(Revenue['revenue'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()

