import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Cases = pd.read_csv('latest_situation_of_reported_cases_covid_19_eng.csv')
Cases_list = []
Reproduction = pd.read_csv('effective_reproductive_local_2.csv')
Repro_list = []

print(Cases['Number of confirmed cases'])
# 24/01/2020 - 06/07/2020
for i in range(16,181):
    Cases_list.append(Cases['Number of confirmed cases'][i])
for i in range(165):
    Repro_list.append(Reproduction['reproductive number'][i])

print(np.corrcoef(Cases_list,Repro_list))