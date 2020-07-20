from pandas import read_csv
from matplotlib import pyplot
series = read_csv('effective_reproductive_local_2.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
series.plot()
pyplot.show()




'''Still doesn't work lol'''
#import pandas as pd
#import matplotlib.pyplot as plt

#series = pd.read_csv('effective_reproductive_local_2.csv')
#plt.figure(figsize=(12.5,4.5))
#plt.plot(series['r0_value'], label='r0 value')
#plt.title('Effective Reproductive Number in Hong Kong (local)')
#plt.xlabel('Time Period')
#plt.ylabel('r0 value')
#plt.legend(loc='upper left')
#plt.show()