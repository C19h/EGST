# author:c19h
# datetime:2022/2/28 13:33
import sys, os, math
import pandas as pd
from chinese_calendar import is_workday
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
import tensorflow.keras as keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# %%
def generate_ele():
    data = pd.read_csv('./data/tyc_127.csv', infer_datetime_format=True)
    data.drop(data[data.q != 0].index, axis=0, inplace=True)
    data["time"] = pd.to_datetime(data["datatime"], infer_datetime_format=True)
    data["time2"] = pd.to_datetime(data["datatime"], infer_datetime_format=True)
    data.set_index("time", inplace=True)
    data.drop(data[data['q'] != 0].index, inplace=True)
    data.drop(["datatime", "_msec", "q"], axis=1, inplace=True)
    power = data.resample("1H").max()
    power['用电量'] = power['val'].shift(-1) - power['val']
    power['时间差'] = power['time2'].shift(-1) - power['time2']
    power.dropna(inplace=True)
    tdiff = pd.Timedelta('0 days 1 hours 10 minutes')
    tdiff2 = pd.Timedelta('0 days 0 hours 50 minutes')
    power = power.drop(power[power['时间差'] > tdiff].index)
    power = power.drop(power[power['时间差'] < tdiff2].index)
    power['is_workday'] = power.index.map(lambda x: is_workday(x))
    power["month"] = power.index.month
    power["hour"] = power.index.hour
    power.drop(power[power['is_workday'] == False].index, inplace=True)
    return power


def generate_temp():
    Files = []
    Files.append('./data/tyc_6079.csv')
    Files.append('./data/tyc_6081.csv')
    Files.append('./data/tyc_6083.csv')
    Files.append('./data/tyc_6085.csv')
    Files.append('./data/tyc_6087.csv')
    Files.append('./data/tyc_6089.csv')
    Files.append('./data/tyc_6091.csv')
    Files.append('./data/tyc_6093.csv')
    main = None
    for i, filename in enumerate(Files):
        data = pd.read_csv(filename, infer_datetime_format=True)
        data.drop(data[data.q != 0].index, axis=0, inplace=True)
        data["time"] = pd.to_datetime(data["datatime"], infer_datetime_format=True)
        data.set_index("time", inplace=True)
        data.drop(data[data['q'] != 0].index, inplace=True)
        data.drop(["datatime", "_msec", "q"], axis=1, inplace=True)
        dmax = data.resample("1H").max()
        dmax.rename(columns={'val': 'max%d' % i}, inplace=True)
        dmin = data.resample("1H").min()
        dmin.rename(columns={'val': 'min%d' % i}, inplace=True)
        if main is None:
            main = pd.merge(dmax, dmin, on='time', how='outer')
        else:
            main = pd.merge(main, dmax, on='time', how='outer')
            main = pd.merge(main, dmin, on='time', how='outer')
    colsmax = ['max%d' % x for x in range(len(Files))]
    colsmin = ['min%d' % x for x in range(len(Files))]
    main['max'] = main[colsmax].mean(axis=1, skipna=True)
    main['min'] = main[colsmin].mean(axis=1, skipna=True)
    for i in range(len(Files)):
        main.drop(['max%d' % i, 'min%d' % i], axis=1, inplace=True)
    main.dropna(inplace=True)
    return main


ele_data = generate_ele()
tem_data = generate_temp()
# %%
ele_data.drop(ele_data[ele_data['is_workday'] == False].index, inplace=True)
data = ele_data.join(tem_data, on='time')
starttime = data.index[0]
# %%
data['delta'] = (data.index - starttime).astype('timedelta64[D]').astype(float)
# %%
X = data[['hour', 'max', 'delta']]
Y = data['用电量']
n_dim = X.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, shuffle=False)
sc = StandardScaler()
x_train = sc.fit_transform(X_train)
x_test = sc.transform(X_test)
# %%
time_step = 12
ahead = 12
x_train_in = np.zeros((len(x_train) - time_step - ahead, time_step, n_dim))
y_train_in = np.zeros((len(y_train) - time_step - ahead, time_step, ahead))
x_test_in = np.zeros((len(x_test) - time_step - ahead, time_step, n_dim))
y_test_in = np.zeros((len(y_test) - time_step - ahead, time_step, ahead))
for i in range(0, len(x_train) - time_step - ahead):
    for j in range(0, time_step):
        x_train_in[i, j, :] = x_train[i + j, :]
        y_train_in[i, j, :] = y_train[i + j:i + j + ahead]
# y_train_in = y_train_in.reshape(-1, time_step, 1)

for i in range(0, len(x_test) - time_step - ahead):
    for j in range(0, time_step):
        x_test_in[i, j, :] = x_test[i + j, :]
        y_test_in[i, j, :] = y_test[i + j:i + j + ahead]


# y_test_in = y_test_in.reshape(-1, time_step, 1)

# %%

def last_time_step_mape(Y_true, Y_pred):
    return keras.metrics.mean_absolute_percentage_error(Y_true[-1, :], Y_pred[-1, :])


def mape(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true)) / np.mean(np.abs(y_true)) * 100


def get_model(n_neurons=50, learning_rate=0.0001):
    inputs = keras.layers.Input(shape=[None, n_dim])
    rnnhiden = keras.layers.LSTM(50, activation='relu', return_sequences=True, dropout=0.2)(inputs)
    output = keras.layers.TimeDistributed(keras.layers.Dense(ahead, activation='relu'))(rnnhiden)
    model = keras.Model(inputs=inputs, outputs=output)
    model.summary()
    model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=[last_time_step_mape])
    return model

rnn = get_model()
checkpoint_cb = keras.callbacks.ModelCheckpoint("./my_zm_model6.h5", save_best_only=True)
rnn.fit(x_train_in, y_train_in, epochs=100,
        validation_data=(x_test_in, y_test_in), batch_size=500,
        callbacks=[checkpoint_cb, keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True)])
# %%
plt.plot(data["用电量"], 'r')
plt.plot(data["max"], 'k')
plt.show()
