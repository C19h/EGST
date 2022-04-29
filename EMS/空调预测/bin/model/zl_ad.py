# -*- coding:utf-8 -*-
# Author:clgh
# 主楼空调用电预测
# ==============================================================================
import sys, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append('./bin/utils')
from csvdataloader import CSVDataLoader
from csvadtemploader import CSVTempLoader
import matplotlib as mpl
import sys, os, math
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
class ZL_ad():
    def __init__(self):
        pass

    def load(self):
        self.PDData = CSVDataLoader('./data/tyc_2987.csv')
        self.PDData.load()
        self.TempData = CSVTempLoader()
        self.TempData.load()

    def get_pddata_workday(self):
        pdday = self.PDData.DayData.drop(self.PDData.DayData[self.PDData.DayData['is_workday'] != 24].index)
        pdday.drop(pdday[pdday['用电量'] <= 10].index, inplace=True)
        return pdday

    def prepare_day_workday(self, pddata):
        pddata.drop(pddata[pddata['is_workday'] == 0].index, inplace=True)
        pddata = pddata.join(self.TempData.TempData, on='time')
        pddata.dropna(axis=1, inplace=True)
        starttime = pddata.index[0]
        pddata['delta'] = (pddata.index - starttime)
        seq0 = np.array(pddata['delta'].astype('timedelta64[D]').astype(float))
        pddata['delta'] = seq0
        # seq1 = np.array(pddata['max'])
        # seq2 = np.array(pddata['用电量'])
        return pddata

    def show_hour_data(self):
        pddata = self.PDData.HourData
        pddata.drop(pddata[pddata['is_workday'] == False].index, inplace=True)
        seq1 = np.array(pddata[pddata['hour'] == 5]['用电量'])
        seq2 = np.array(pddata[pddata['hour'] == 6]['用电量'])
        seq3 = np.array(pddata[pddata['hour'] == 7]['用电量'])
        seq4 = np.array(pddata[pddata['hour'] == 8]['用电量'])
        # self._plot([seq1, seq2, seq3, seq4])

    def show_day_data(self):
        pddata = self.PDData.DayData
        # pddata.drop(pddata[pddata['is_workday'] == False].index, inplace=True)
        seq1 = np.array(pddata[pddata['hour'] == 5]['用电量'])
        seq2 = np.array(pddata[pddata['hour'] == 6]['用电量'])
        seq3 = np.array(pddata[pddata['hour'] == 7]['用电量'])
        seq4 = np.array(pddata[pddata['hour'] == 8]['用电量'])
        # self._plot([seq1, seq2, seq3, seq4])
        seq5 = np.array(pddata["用电量"])
        fig = plt.figure(figsize=(24, 12))  # 调整画图空间的大小
        plt.plot(pddata.index, pddata["用电量"], linestyle='', marker='.', markersize=10, c='r')  # 作图
        ax = plt.gca()
        date_format = mpl.dates.DateFormatter('%Y-%m-%d')  # 设定显示的格式形式
        ax.xaxis.set_major_formatter(date_format)  # 设定x轴主要格式
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(30))  # 设定坐标轴的显示的刻度间隔
        fig.autofmt_xdate()  # 防止x轴上的数据重叠，自动调整。
        plt.show()
        # plt.plot(seq5)
        # plt.show()

    def _plot(self, npdata):
        for i, r in enumerate(npdata):
            plt.plot(r, label='data%d' % (i + 1))
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # %%
    ad = ZL_ad()
    ad.load()
    wdata = ad.get_pddata_workday()
    pddata = ad.prepare_day_workday(wdata)
    pddata["month"] = pddata.index.month


    # ad.show_hour_data()
    # ad.show_day_data()
    def plot(all):
        fig = plt.figure(figsize=(24, 12))  # 调整画图空间的大小
        ax1 = fig.add_subplot(111)
        ax1.plot(all.index, all["用电量"], linestyle='', marker='.', markersize=10, c='r', label='val')  # 作图
        ax1.legend(loc='upper left')
        ax2 = ax1.twinx()
        ax2.plot(all.index, all["mean"], linestyle='', marker='*', markersize=10, c='k', label='temp')  # 作图
        ax2.legend(loc='upper center')
        ax = plt.gca()
        date_format = mpl.dates.DateFormatter('%Y-%m-%d')  # 设定显示的格式形式
        ax.xaxis.set_major_formatter(date_format)  # 设定x轴主要格式
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(20))  # 设定坐标轴的显示的刻度间隔
        fig.autofmt_xdate()  # 防止x轴上的数据重叠，自动调整。
        plt.show()


    plot(pddata)
    # %%
    X_train, X_test, y_train, y_test = train_test_split(pddata[['mean', 'month', '用电量']], pddata['用电量'],
                                                        test_size=0.2,
                                                        shuffle=False)
    scaler = StandardScaler()
    scy = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    y_train_s = scy.fit_transform(np.array(y_train)[:, np.newaxis])
    y_test_s = scy.transform(np.array(y_test)[:, np.newaxis])
    n_dim = X_train.shape[1]
    # %%
    time_step = 5
    x_train_all = np.zeros((len(X_train_s) - time_step, time_step, n_dim))
    y_train_all = np.zeros((len(y_train_s) - time_step, time_step))
    x_test_all = np.zeros((len(X_test_s) - time_step, time_step, n_dim))
    y_test_all = np.zeros((len(y_test_s) - time_step, time_step))
    for i in range(0, len(X_train_s) - time_step):
        x_train_all[i, :, :] = X_train_s[i:i + time_step, :]
        y_train_all[i, ...] = y_train_s[i + 1:i + time_step + 1, 0]
    # y_train_all = y_train_all.reshape(-1, time_step, 1)

    for i in range(0, len(X_test_s) - time_step):
        x_test_all[i, :, :] = X_test_s[i:i + time_step, :]
        y_test_all[i, ...] = y_test_s[i + 1:i + time_step + 1, 0]


    # y_test_all = y_test_all.reshape(-1, time_step, 1)

    # %%
    def last_time_step_mape(Y_true, Y_pred):
        return keras.metrics.mean_absolute_percentage_error(Y_true[:, -1], Y_pred[:, -1])


    def mape(y_true, y_pred):
        return np.mean(np.abs(y_pred - y_true)) / np.mean(np.abs(y_true)) * 100


    def get_model(n_neurons=50, learning_rate=0.001):
        inputs = keras.layers.Input(shape=[None, n_dim])
        rnnhiden = keras.layers.SimpleRNN(50, activation=keras.layers.LeakyReLU(), return_sequences=True)(inputs)
        BN = keras.layers.BatchNormalization()(rnnhiden)
        rnnhiden = keras.layers.SimpleRNN(20, activation=keras.layers.LeakyReLU(), return_sequences=True)(BN)
        output = keras.layers.TimeDistributed(keras.layers.Dense(1, activation=keras.layers.LeakyReLU()))(rnnhiden)
        model = keras.Model(inputs=inputs, outputs=output)
        model.summary()
        model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      metrics=[last_time_step_mape])
        return model


    tf.random.set_seed(42)
    is_train = False
    if is_train:
        rnn = get_model()
        checkpoint_cb = keras.callbacks.ModelCheckpoint("./my_zm_model.h5", save_best_only=True)
        rnn.fit(x_train_all, y_train_all, epochs=500,
                validation_data=(x_test_all, y_test_all), batch_size=60,
                callbacks=[checkpoint_cb, keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)])
        # =========================================================================================================================
        # keras_reg = keras.wrappers.scikit_learn.KerasRegressor(get_model)
        # param = {
        #     "n_neurons": [20, 25, 30, 35, 40],
        #     "learning_rate": reciprocal(1e-4, 1e-3)
        #
        # checkpoint_cb = keras.callbacks.ModelCheckpoint("./my_zm_model1.h5", save_best_only=True)
        # rnd = RandomizedSearchCV(keras_reg, param, n_iter=5, cv=3)
        # history = rnd.fit(x_train, y_train, epochs=200,
        #                   validation_data=(x_test, y_test), batch_size=40,
        #                   callbacks=[checkpoint_cb, keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)])
        # =========================================================================================================================
        pre_now = rnn.predict(x_test_all)
        print("now_model_pre:", np.mean(last_time_step_mape(y_test_all, pre_now)))
    # %%
    # saved = keras.models.load_model('my_zm_model.h5', compile=False)
    # pre = saved.predict(x_test_all)
    # print("saved_best_model_pre:", np.mean(last_time_step_mape(y_test_all, pre)))

    # 前一天做后一天的预测值
    # data = ad.PDData.HourData
    # data.drop(data[data["is_workday"] == 0].index, inplace=True)
    # bbb = pd.merge(data['用电量'].shift(1), data['用电量'], on='time')
    # bbb = bbb.dropna()
    # print(mape(np.array(bbb['用电量_x']), np.array(bbb['用电量_y'])))
