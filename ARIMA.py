# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 08:12:11 2018

@author: Ashtami
"""
import os
os.chdir("C:/Users/Ashtami/Documents/Python/")

from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
#------ Pre-Procesing of imported Dataset with Pandas ---------#
df_all = pd.read_csv('GlobalLandTemperaturesByCountry.csv', header=0)
# Dropping 'AvergeTemperatureUncertainty' column-This column is useless for our case !!
df_all_reduced = df_all.drop('AverageTemperatureUncertainty', axis=1)
# Filtering 'France' as country
df_france = df_all_reduced [df_all_reduced .Country == 'France']
# Dropping 'Country' column
df_france = df_france.drop('Country', axis=1)
# Converting 'Date' column to a datetime format index to access data based on dates.
df_france.index = pd.to_datetime(df_france['Date'])
# dropping 'Date' column. We use dates as index from now on, so we don't need them as an extra column(input)
df_france = df_france.drop('Date', axis=1)
# Filtering data starting from 1950-01-01
df_france = df_france.ix['1960-01-01':]
# Sorting index in an ascending way.
df_france = df_france.sort_index()
# Replacing 'NaN' values with the last valid observation
df_france.AverageTemperature.fillna(method='pad', inplace=True)
# Extract Out the Timeseries values part
timeseries = df_france.AverageTemperature
#----------------------------- ARIMA ---------------------------------------
size = int(len(timeseries) - 9)
train, test = timeseries[0:size], timeseries[size:len(timeseries)]
previous_samples = [x for x in train]
for t in range(len(test)):
 model = ARIMA(previous_samples, order=(10, 0, 1))
 model_fit = model.fit(disp=0)
 output = model_fit.forecast()
 yhat = output[0] #get the first element which is the forecast, we don't need the rest
 obs = test[t]
 previous_samples.append(obs)
 print('predicted=%f, expected=%f' % ((yhat), (obs)))
 
 
 ###############EXAMPLE 2#################
 from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import numpy as np
from datetime import datetime
# ============ Creating a random timeseries ========== #
counts= np.arange(1, 21) + 0.2 * (np.random.random(size=(20,)) - 0.5)
start = pd.datetime.strptime("1 Nov 16", "%d %b %y")
daterange = pd.date_range(start, periods=20)
table = {"count": counts, "date": daterange}
# ================= Pre-processing ====================#
data = pd.DataFrame(table)
data.set_index("date", inplace=True)
print(data)
# =============== Setting up ARIMA model ============= #
model = ARIMA(data[0:len(data)-1], (1,1,1))
model_fit = model.fit(disp=0)
print(model_fit.forecast())
 
 
 