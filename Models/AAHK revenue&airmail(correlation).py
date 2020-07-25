import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Data
Revenue = pd.read_csv('/Users/william_whf/PycharmProjects/PwC-stem-team-headache-/Data/AAHK_revenue.csv')
Revenue_ = []
airmail = pd.read_csv('/Users/william_whf/PycharmProjects/PwC-stem-team-headache-/Data/Cargo and Airmail.csv')
airmail_ = []

#Correlation
for i in range(Revenue['revenue'].size):
    Revenue_.append(Revenue['revenue'][i])
    airmail_.append(airmail['tonnes'][i])

print(np.corrcoef(Revenue_,airmail_))

#Visualize the data
fig, ax1 = plt.subplots()

color = 'tab:red'
plt.title('Relationship Between Cargo & Airmail and Revenue of AAHK')
ax1.set_xlabel('Time Period(Yearly)')
ax1.set_ylabel('Cargo & Airmail(millions of tonnes)', color=color)
ax1.plot(airmail['tonnes'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('Revenue(in HK$ million)', color=color)
ax2.plot(Revenue['revenue'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()

