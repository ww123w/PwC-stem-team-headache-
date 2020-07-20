from pandas import read_csv
from matplotlib import pyplot
series = read_csv('effective_reproductive_local_2.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
x = series.to_list()
y = range(0,165)
pyplot.plot(y,x)
pyplot.show()