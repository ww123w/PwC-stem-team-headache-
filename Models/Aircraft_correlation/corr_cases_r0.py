import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir("../../Data")

Cases = pd.read_csv('confirmed_cases_daily.csv')
Cases_list = []
Reproduction = pd.read_csv('reproductive_number.csv')
Repro_list = []

# 24/01/2020 - 06/07/2020
for i in range(16,181):
    Cases_list.append(Cases['Number of confirmed cases'][i])
for i in range(165):
    Repro_list.append(Reproduction['reproductive number'][i])

print(np.corrcoef(Cases_list,Repro_list))