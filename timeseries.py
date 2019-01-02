# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 07:28:42 2018

@author: Ashtami
"""

import os
os.chdir("C:/Users/Ashtami/Documents/Python/")

from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
import matplotlib.pylab as plt
import pandas as pd
#----------- Pre-Procesing of Timeseries with Pandas --------------#
data_frame = pd.read_csv('Airpassenger.csv', header=0)
data_frame ['Month'] = pd.to_datetime(data_frame ['Month'])
indexed_df = data_frame .set_index('Month')
timeseries = indexed_df['Passengers']
#--------------------- Some Timeseries Analysis -------------------#
lag_acf = acf(timeseries, nlags=20)
lag_pacf = pacf(timeseries, nlags=20)
rolmean = pd.rolling_mean(timeseries, window=12)
rolstd = pd.rolling_std(timeseries, window=12)
#------------------------- Plotting -------------------------------#
plt.subplot(221)
plt.plot(timeseries, color='black', label='original')
plt.plot(rolmean, color='blue', label='Rolling Mean')
plt.plot(rolstd, color='red', label='Rolling Deviation')
plt.legend(loc='best')
plt.title('Original Data, Rolling Mean & Standard Deviation')

plt.subplot(223)
plt.plot(lag_pacf, color='orange', label='auto correlation func')
plt.legend(loc='best')
plt.title('Partial Auto Correlation Function')

plt.subplot(224)
plt.plot(lag_acf, color='green', label='partial auto correlation func ')
plt.legend(loc='best')
plt.title('Auto Correlation Function')
plt.show()


##############EXAMPLE 2#####################
# load dataset
df = pd.read_csv('c:/Users/MA/Desktop/TimeSer.csv', header=0)
df['Date'] = pd.to_datetime(df['Date'])
indexed_df = df.set_index('Date')
timeseries = indexed_df['Value']
split_point = len(timeseries) - 10
dataset, validation = timeseries[0:split_point], timeseries[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
#def difference(dataset, interval=1):
# diff = list()
# for i in range(interval, len(dataset)):
# value = dataset[i] - dataset[i - interval]
# diff.append(value)
# return np.array(diff)
#X = dataset.values
#months_in_year = 12
#differenced = difference(X, months_in_year)
#fit model
history = [x for x in dataset]
predictions = list()
print('Printing Predicted vs Expected Values...')
print('\n')
for t in range(len(validation)):
 model = ARIMA(history, order=(2,1,1))
 model_fit = model.fit(disp=0)
 output = model_fit.forecast()
 yhat = output[0]
 predictions.append(float(yhat))
 obs = validation[t]
 history.append(obs)
 print('predicted=%f, expected=%f' % ((yhat), (obs)))