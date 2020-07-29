import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

os.chdir("../Data")
data = pd.read_csv("aircraft_2000_2020.csv")

data = data.drop(columns = ["Year","Landing", "Take-off", "Total", "Year-on-year \r\n% change", "Arrival", "Departure", "Year-on-year\r\n% change", "Unloaded",
                            "Loaded", "Total.1", "Year-on-year \r\n% change.1"])
# adding two features
data["month_index"] = data.index + 1
data["month_index_sqr"] = data["month_index"] ** 2

# Converting months to indicators
data = pd.get_dummies(data, columns = ["Month"])
print(data)

# Partition the data
# 2003 SARS & 2008 Financial crisis -> noise. Therefore, training & validation from 2010-01-01 (index 120) to 2018-12-31 (index 227) (108 months)
# Training: 2010-01-01 (index 120) to 2016-12-31 (index 203) (7 years / 84 months)
train = data.loc[120:203]
# Validation : 2017-01-01 (index 204) to 2018-12-31 (index 227) (2 years / 24 months)
valid = data.loc[204:227]

train_valid = data.loc[120:227]

forecast = data.loc[252:263]

# Taking away output variable
output_train = train["Total_"]
train = train.drop(columns = ["Total_"])
print(output_train)
print(train)