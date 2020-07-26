import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
#Data

os.chdir("../../Data")
Revenue = pd.read_csv('AAHK_revenue.csv')
Revenue_ = []
airmail = pd.read_csv('Cargo and Airmail.csv')
airmail_ = []

#Correlation
for i in range(Revenue['revenue'].size):
    Revenue_.append(Revenue['revenue'][i])
    airmail_.append(airmail['tonnes'][i])

print(np.corrcoef(Revenue_,airmail_))

#Visualize the data
fig, ax1 = plt.subplots()

color = 'tab:red'
plt.title('Relationship Between Revenue of AAHK and Cargo & Airmail')
ax1.set_ylabel('Cargo & Airmail(millions of tonnes)', color=color)
ax1.plot(airmail['tonnes'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_xticklabels([1998,2000,2003,2006,2009,2011,2013,2015,2017,2020])
ax2.set_ylabel('Revenue(in HK$ million)', color=color)
ax2.plot(Revenue['revenue'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()

