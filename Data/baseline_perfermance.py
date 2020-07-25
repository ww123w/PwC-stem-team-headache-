import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# load dataset
series = pd.read_csv('HK_aircraft_til2020Jun_cleaned2.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
aircraft = series['Total_']
aircraft.index = pd.date_range(start='2000-1-01', end = '2020-07-01', freq='M')

def univariate_data(dataset, start_index, end_index, history_size, target_size):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i)
    # Reshape data from (history_size,) to (history_size, 1)
    data.append(np.reshape(dataset[indices], (history_size, 1)))
    labels.append(dataset[i+target_size])
  return np.array(data), np.array(labels)

# 2000-1-1 - 2017-12-31 for training
TRAIN_SPLIT = 12*18
data_mean = aircraft[:TRAIN_SPLIT].mean()
data_std = aircraft[:TRAIN_SPLIT].std()

print(data_mean)
print(data_std)
#standarization
aircraft = (aircraft.values - data_mean) / data_std

# Prediction
past_history = 12
future_target = 0
x_train, y_train = univariate_data(aircraft, 0, TRAIN_SPLIT, past_history, future_target)
x_val, y_val = univariate_data(aircraft, TRAIN_SPLIT, None, past_history, future_target)

def create_time_steps(length):
  return list(range(-length, 0))

def show_plot(plot_data, delta, title):
  labels = ['History', 'True Future', 'Model Prediction']
  marker = ['.-', 'rx', 'go']
  time_steps = create_time_steps(plot_data[0].shape[0])
  if delta:
    future = delta
  else:
    future = 0

  plt.title(title)
  for i, x in enumerate(plot_data):
    if i:
      plt.plot(future, plot_data[i], marker[i], markersize=10,
               label=labels[i])
    else:
      plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
  plt.legend()
  plt.xlim([time_steps[0], (future+5)*2])
  plt.xlabel('Time-Step')
  return plt

def baseline(history):
  return np.mean(history)

show_plot([x_train[0], y_train[0], baseline(x_train[0])], 0, 'Baseline Prediction Example')
plt.show()