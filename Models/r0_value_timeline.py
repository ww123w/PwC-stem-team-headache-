from pandas import read_csv
from matplotlib import pyplot
import os
os.chdir("../Data")

series = read_csv('reproductive_number.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
x = series.to_list()
y = range(0,165)
pyplot.plot(y,x)
pyplot.show()