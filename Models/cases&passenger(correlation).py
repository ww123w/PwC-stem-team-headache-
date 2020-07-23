import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Data
Passenger = pd.read_csv('/Users/william_whf/PycharmProjects/PwC-stem-team-headache-/Data/aircraft.csv')
Passenger_ = []
Confirmed_cases = pd.read_csv('/Users/william_whf/PycharmProjects/PwC-stem-team-headache-/Data/confirmed cases(monthly).csv')
Confirmed_cases_ = []

#Visualize the data
fig, ax1 = plt.subplots()

color = 'tab:red'
plt.title('Relationship Between No. of Confirmed Cases and Total No. of Passengers')
ax1.set_xlabel('Time Period')
ax1.set_ylabel('No. of Confirmed Cases', color=color)
ax1.plot(Confirmed_cases['Number of confirmed cases'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('Total Number of Passengers', color=color)
ax2.plot(Passenger['Total_'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()

for i in range(Passenger['Total_'].size):
    Passenger_.append(Passenger['Total_'][i])
    Confirmed_cases_.append(Confirmed_cases['Number of confirmed cases'][i])

print(np.corrcoef(Confirmed_cases_,Passenger_))