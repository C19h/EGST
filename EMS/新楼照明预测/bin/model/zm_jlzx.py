#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author:huxiao
# 国际技术交流中心照明用电
# ==============================================================================
# %%
import sys, os, math
import pandas as pd
sys.path.append('./bin/utils')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
from csvdataloader import CSVDataLoader
from csvtemploader import CSVTempLoader
import tensorflow.keras as keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


# %%

class ZM_Jlzx():
    def __init__(self):
        pass

    def load(self):
        self.PDData = CSVDataLoader('./data/tyc_127.csv')
        self.PDData.load()
        self.PDTemp = CSVTempLoader()
        self.PDTemp.load()

    def show_hour_data(self):
        pddata = self.PDData.HourData
        pddata.drop(pddata[pddata['is_workday'] == 0].index, inplace=True)
        seq1 = np.array(pddata[pddata['hour'] == 5]['用电量'])
        seq2 = np.array(pddata[pddata['hour'] == 6]['用电量'])
        seq3 = np.array(pddata[pddata['hour'] == 7]['用电量'])
        seq4 = np.array(pddata[pddata['hour'] == 8]['用电量'])
        self._plot([seq1, seq2, seq3, seq4])

    def get_pddata_workday(self):
        pdday = self.PDData.DayData.drop(self.PDData.DayData[self.PDData.DayData['is_workday'] != 24].index)
        pdday.drop(pdday[pdday['用电量'] <= 220].index, inplace=True)
        return pdday

    def prepare_day_workday(self, pddata):
        pddata.drop(pddata[pddata['is_workday'] == 0].index, inplace=True)
        pddata = pddata.join(self.PDTemp.DayData, on='time')
        pddata.dropna(inplace=True)
        starttime = pddata.index[0]
        pddata['delta'] = (pddata.index - starttime)
        seq0 = np.array(pddata['delta'].astype('timedelta64[D]').astype(float))
        pddata['delta'] = seq0
        # seq1 = np.array(pddata['max'])
        # seq2 = np.array(pddata['用电量'])
        return pddata

    def train1(self, seq0, seq1, seq2):
        self.x1 = self.solute_temp(seq0, seq1, seq2)

    def predict1(self, x1, x2):
        seq4 = self._fx2(x1, x2)
        # self._plot([seq4 + seq2, seq2])
        return seq4

    def _fx2(self, x1, x2):
        a = self.x1[0]
        c = self.x1[1]
        d = self.x1[2]
        e = self.x1[3]
        # x: time, temp, power
        ans = (np.log(a * x1 + 1000) + c) * d * x2 + 800 * e
        return ans

    def _fx(self, x1):
        a = x1[0]
        c = x1[1]
        d = x1[2]
        e = x1[3]
        # x: time, temp, power
        ans = (np.log(a * self.x[0, :] + 1000) + c) * d * self.x[1, :] - self.x[2, :] + 800 * e
        return ans

    def _fx_call(self, x1):
        res = self._fx(x1)
        ans = np.sum(np.sqrt(res * res))
        return ans if not np.isnan(ans) else 9999999999

    def solute_temp(self, time_arr, temp_arr, powerarr):
        self.x = np.array([time_arr, temp_arr, powerarr])
        x0 = [1, -8.21, -10, 0.5]
        res = minimize(self._fx_call, x0)
        print(res.fun, res.x)
        return res.x

    def _plot(self, npdata):
        for i, r in enumerate(npdata):
            plt.plot(r, label='data%d' % (i + 1))
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # %%
    zj = ZM_Jlzx()
    zj.load()
    wdata = zj.get_pddata_workday()
    pddata = zj.prepare_day_workday(wdata)
    # 模型1
    # x0_1 = x0_1.reshape(-1, 1)
    # x0_2 = x0_2.reshape(-1, 1)
    # X = np.concatenate([x0_1, x0_2], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(pddata[['max', 'delta']], pddata['用电量'], test_size=0.2,
                                                        shuffle=False)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    # %%
    time_step = 7
    x_train_all = np.zeros((len(X_train_s) - time_step, time_step, 2))
    y_train_all = np.zeros((len(y_train) - time_step, time_step))
    x_test_all = np.zeros((len(X_test_s) - time_step, time_step, 2))
    y_test_all = np.zeros((len(y_test) - time_step, time_step))
    for i in range(0, len(X_train_s) - time_step):
        x_train_all[i, :, :] = X_train_s[i:i + time_step, :]
        y_train_all[i, ...] = y_train[i:i + time_step]
    y_train_all = y_train_all.reshape(-1, time_step, 1)

    for i in range(0, len(X_test_s) - time_step):
        x_test_all[i, :, :] = X_test_s[i:i + time_step, :]
        y_test_all[i, ...] = y_test[i:i + time_step]
    y_test_all = y_test_all.reshape(-1, time_step, 1)


    # %%
    def last_time_step_mape(Y_true, Y_pred):
        return keras.metrics.mean_absolute_percentage_error(Y_true[:, -1], Y_pred[:, -1])


    def mape(y_true, y_pred):
        return np.mean(np.abs(y_pred - y_true)) / np.mean(np.abs(y_true)) * 100


    def get_model(n_neurons=50, learning_rate=0.01):
        inputs = keras.layers.Input(shape=[None, 2])
        rnnhiden = keras.layers.LSTM(100, activation=keras.layers.LeakyReLU(), return_sequences=True)(inputs)
        BN = keras.layers.BatchNormalization()(rnnhiden)
        rnnhiden = keras.layers.LSTM(100, activation=keras.layers.LeakyReLU(), return_sequences=True)(BN)
        BN = keras.layers.BatchNormalization()(rnnhiden)
        output = keras.layers.TimeDistributed(keras.layers.Dense(1, activation=keras.layers.LeakyReLU()))(BN)
        model = keras.Model(inputs=inputs, outputs=output)
        model.summary()
        model.compile(loss="mse", optimizer=keras.optimizers.Nadam(learning_rate=learning_rate, decay=0.01),
                      metrics=[last_time_step_mape])
        return model


    # %%
    tf.random.set_seed(42)
    is_train = False
    if is_train:
        rnn = get_model()
        checkpoint_cb = keras.callbacks.ModelCheckpoint("./my_zm_model6.h5", save_best_only=True)
        rnn.fit(x_train_all, y_train_all, epochs=250,
                validation_data=(x_test_all, y_test_all), batch_size=50,
                callbacks=[checkpoint_cb, keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)])
        # =========================================================================================================================
        # keras_reg = keras.wrappers.scikit_learn.KerasRegressor(get_model)
        # param = {
        #     "n_neurons": [20, 25, 30, 35, 40],
        #     "learning_rate": reciprocal(1e-4, 1e-3)
        # }
        # checkpoint_cb = keras.callbacks.ModelCheckpoint("./my_zm_model1.h5", save_best_only=True)
        # rnd = RandomizedSearchCV(keras_reg, param, n_iter=5, cv=3)
        # history = rnd.fit(x_train, y_train, epochs=200,
        #                   validation_data=(x_test, y_test), batch_size=40,
        #                   callbacks=[checkpoint_cb, keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)])
        # =========================================================================================================================
        # %%
        pre_now = rnn.predict(x_test_all)
        print("now_model_pre:", np.mean(last_time_step_mape(y_test_all, pre_now)))


    # %%
    saved = keras.models.load_model('11%.h5', compile=False)
    pre = saved.predict(x_test_all)
    print("saved_best_model_pre:", np.mean(last_time_step_mape(y_test_all, pre)))

    def get_mape_with_hour(pre):
        pre_last = pre[:, -1, :]
        pre_last = pd.DataFrame(pre_last, columns=['预测用电'],
                                index=list(
                                    pd.to_datetime(y_test.index[-(len(y_test) - time_step):], format='%Y-%m-%d')))
        a = zj.PDData.HourData
        daylist = [x.strftime('%Y-%m-%d') for x in pre_last.index]
        a['tag_remove'] = a.index.map(lambda x: False if x.strftime('%Y-%m-%d') in daylist else True)
        a.drop(a[a['tag_remove'] == True].index, inplace=True)
        a.drop(["tag_remove"], axis=1, inplace=True)
        a['该天预测电量'] = a.index.map(lambda x: pre_last.loc[x.strftime('%Y-%m-%d')]['预测用电'])
        a['每小时预测用电'] = a['日用电比'].shift(24) * a['该天预测电量']
        a.dropna(inplace=True)
        return mape(np.array(a['每小时预测用电']), np.array(a['用电量'])), mape(np.array(a['用电量'][:-24]), np.array(a['用电量'][24:]))


    # %%
    # plt.plot(y_test_all[:, -1, 0], 'r', label="true")
    # plt.plot(pre[:, -1, 0], 'k', label="pre")
    # plt.legend()
    # plt.show()
    print(get_mape_with_hour(pre))
    # %%
    # data = zj.PDData.HourData
    # data.drop(data[data["is_workday"] == 0].index, inplace=True)
    # aaa = data['用电量'].shift(24).dropna()
    # bbb = pd.merge(data['用电量'].shift(24), data['用电量'], on='time')
    # bbb = bbb.dropna()
    # print(mape(np.array(bbb['用电量_x']), np.array(bbb['用电量_y'])))
