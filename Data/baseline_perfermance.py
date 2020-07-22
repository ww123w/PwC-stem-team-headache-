import pandas as pd
from matplotlib import pyplot

# load dataset

series = pd.read_csv('HK_aircraft_til2020Jun_cleaned2.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
print(series)

series.plot()
pyplot.show()