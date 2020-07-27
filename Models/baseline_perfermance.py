import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# load dataset
os.chdir("../Data")
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

#standarization
aircraft = (aircraft.values - data_mean) / data_std

# Prediction
past_history = 20
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

show_plot([x_train[0], y_train[0], baseline(x_train[0])], 0, 'Baseline Prediction')
#plt.show()

batch = 256
buffer_size = 10000

train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train = train.cache().shuffle(buffer_size).batch(batch).repeat()

val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val = val.batch(batch).repeat()

lstm = tf.keras.models.Sequential([ tf.keras.layers.LSTM(8, input_shape=x_train.shape[-2:]), tf.keras.layers.Dense(1)])

lstm.compile(optimizer='adam', loss='mae')

# prediction test
for x, y in val.take(1):
    print(lstm.predict(x).shape)

EVALUATION_INTERVAL = 200
EPOCHS = 10

lstm.fit(train, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL, validation_data=val, validation_steps=50)

for x, y in val.take(3):
  plot = show_plot([x[0].numpy(), y[0].numpy(), lstm.predict(x)[0]], 0, 'LSTM model')
  plot.show()

#Part 2

features_considered = ['Total_']
features = series[features_considered]
features.index = series['Month']
features.head()

features.plot(subplots=True)

dataset = features.values
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
data_std = dataset[:TRAIN_SPLIT].std(axis=0)
dataset = (dataset-data_mean)/data_std

#Single step model

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)

past_history = 246
future_target = 12
STEP = 6    #idk

x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 1], 0,   #Error
                                                   TRAIN_SPLIT, past_history,
                                                   future_target, STEP,
                                                   single_step=True)
x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 1],
                                               TRAIN_SPLIT, None, past_history,
                                               future_target, STEP,
                                               single_step=True)

train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
train_data_single = train_data_single.cache().shuffle(buffer_size).batch(batch).repeat()

val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
val_data_single = val_data_single.batch(batch).repeat()

single_step_model = tf.keras.models.Sequential()
single_step_model.add(tf.keras.layers.LSTM(32,
                                           input_shape=x_train_single.shape[-2:]))
single_step_model.add(tf.keras.layers.Dense(1))

single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS,
                                            steps_per_epoch=EVALUATION_INTERVAL,
                                            validation_data=val_data_single,
                                            validation_steps=50)

def plot_train_history(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()

  plt.show()

plot_train_history(single_step_history,
                   'Single Step Training and validation loss')