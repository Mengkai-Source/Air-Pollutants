import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
import math
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
from keras import losses
from numpy import hstack

## Univariate prediction
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

df = pd.read_csv('/Users/mengkaixu/Desktop/AirQualityUCI/AirQualityUCI1.csv')
df.drop('Unnamed: 15', axis=1, inplace=True)
df.drop('Unnamed: 16', axis=1, inplace=True)
df = df[:-114]
df[df.columns] = df[df.columns].replace({-200:np.nan})
df = df.fillna(df.mean())
train, test = 0.7, 0.3
value = df.values

# ARIMA Univariate
from pandas.plotting import autocorrelation_plot
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

autocorrelation_plot(df['NO2(GT)'])
# fit model
model = ARIMA(df['NO2(GT)'][0:math.floor(len(df['NO2(GT)'])*train)], order=(5,0,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
model_fit.forcast

X = df['NO2(GT)'].values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()


GT_NO2 = value[:,9]
n_steps = 8
X, y = split_sequence(GT_NO2, n_steps)
X_train, y_train = X[:math.floor(len(X[:,0])*train),:], y[:math.floor(len(y)*train)]
X_test, y_test = X[math.floor(len(X[:,0])*train):,:], y[math.floor(len(y)*train):]
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(50, activation='relu',return_sequences=True))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))  # Output layer 1 node
model.compile(optimizer='adam', loss=losses.mean_absolute_percentage_error)
# fit model
model.fit(X_train, y_train, epochs=20, verbose=0)

X_test = X_test.reshape((X_test.shape[0], X_train.shape[1], n_features))
yhat = model.predict(X_test, verbose=0)
a = 0
for i in range(len(yhat)):
    a += abs(yhat[i] - y_test[i])/y_test[i]
MRE = a/len(yhat)
print('Error ratio of Univariate %.3f' % MRE)

# Best so far stacked LSTM (3-layer)


## Multivariate Prediction
def split_sequences(sequences, n_steps):
    X = list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x = sequences[i:end_ix, :]
        X.append(seq_x)
    return array(X)

CO_pt = value[:,3].reshape((len(value[:,3]), 1))
NMHC_pt = value[:,6].reshape((len(value[:,6]), 1))
NOx_pt = value[:,8].reshape((len(value[:,8]), 1))
NO2_pt = value[:,10].reshape((len(value[:,10]), 1))
O3_pt = value[:,11].reshape((len(value[:,11]), 1))
dataset = hstack((CO_pt, NMHC_pt, NOx_pt, NO2_pt, O3_pt))
n_steps = 4
X = split_sequences(dataset, n_steps)
n_features = X.shape[2]
y = value[:,9][n_steps-1:]
# Split data into Training and Validating datasets
X_train, y_train = X[:math.floor(len(X[:,0])*train),:], y[:math.floor(len(y)*train)]
X_test, y_test = X[math.floor(len(X[:,0])*train):,:], y[math.floor(len(y)*train):]


model = Sequential()
model.add(Bidirectional(LSTM(60, activation='relu'), input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss=losses.mean_absolute_percentage_error)
# fit model
model.fit(X_train, y_train, epochs=100, verbose=0)
X_test = X_test.reshape((X_test.shape[0], X_train.shape[1], X_train.shape[2]))
yhat = model.predict(X_test, verbose=0)
a = 0
for i in range(len(yhat)):
    a += abs(yhat[i] - y_test[i])/y_test[i]
MRE = a/len(yhat)
print('Error ratio of Multivariate %.3f' % MRE)


from keras.models import Sequential
from keras.layers import Dense
n_input = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape((X_train.shape[0], n_input))
model = Sequential()
model.add(Dense(60, activation='relu',input_dim=n_input))
#model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss=losses.mean_absolute_percentage_error)
# fit model
model.fit(X_train, y_train, epochs=200, verbose=0)
# demonstrate prediction
X_test = X_test.reshape((X_test.shape[0], n_input))
yhat = model.predict(X_test, verbose=0)
a = 0
for i in range(len(yhat)):
    a += abs(yhat[i] - y_test[i])/y_test[i]
MRE = a/len(yhat)
print('Error ratio of Multivariate %.3f' % MRE)