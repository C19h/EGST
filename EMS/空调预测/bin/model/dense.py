# author:c19h
# datetime:2022/3/2 16:44
import sys, os
sys.path.append('./bin/utils')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.append('./bin/utils')
from csvdataloader import CSVDataLoader
from csvadtemploader import CSVTempLoader
import matplotlib as mpl
from sklearn.model_selection import cross_val_score, GridSearchCV
import sys, os, math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_percentage_error
import tensorflow.keras as keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def mape(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true) / (np.abs(y_true)))*100

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
        ax1.plot(all.index, all["用电量"], linestyle='', marker='.', markersize=10, c='r', label='用电')  # 作图
        ax2 = ax1.twinx()
        ax2.plot(all.index, all["mean"], linestyle='', marker='*', markersize=10, c='k', label='temp')  # 作图
        ax = plt.gca()
        date_format = mpl.dates.DateFormatter('%Y-%m-%d')  # 设定显示的格式形式
        ax.xaxis.set_major_formatter(date_format)  # 设定x轴主要格式
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(20))  # 设定坐标轴的显示的刻度间隔
        fig.autofmt_xdate()  # 防止x轴上的数据重叠，自动调整。
        plt.legend()
        plt.show()


    # plot(pddata)
    # %%
    X_train, X_test, y_train, y_test = train_test_split(pddata[['mean', 'month', '用电量']], pddata['用电量'],
                                                        test_size=0.2,
                                                        shuffle=False)
    scaler = StandardScaler()
    scy = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    # y_train_s = scy.fit_transform(np.array(y_train)[:, np.newaxis])
    # y_test_s = scy.transform(np.array(y_test)[:, np.newaxis])
    time_step = 3
    n_dim = X_train.shape[1]
    x_train_all = np.zeros((len(X_train_s) - time_step, n_dim * time_step))
    y_train_all = np.zeros((len(y_train) - time_step, 1))
    x_test_all = np.zeros((len(X_test_s) - time_step, n_dim * time_step))
    y_test_all = np.zeros((len(y_test) - time_step, 1))
    for i in range(0, len(X_train_s) - time_step):
        x_train_all[i, :] = np.array(X_train_s[i:i + time_step, :]).reshape(-1)
        y_train_all[i, :] = np.array(y_train[i + time_step])

    for i in range(0, len(X_test_s) - time_step):
        x_test_all[i, :] = np.array(X_test_s[i:i + time_step, :]).reshape(-1)
        y_test_all[i, :] = np.array(y_test[i + time_step])


    # timestep = 5
    # x_train_in = np.zeros((len(X_train) - timestep, n_dim * timestep))
    # y_train_in = np.zeros((len(X_train) - timestep, 1))
    # for i in range(0, len(X_train) - timestep):
    #     x_train_in[i, :] = np.array(X_train_s[i:i + timestep, :]).reshape(-1)
    #     y_train_in[i, :] = np.array(y_train_s[])
    # %%
    class MyDenseModel(keras.Model):
        def __init__(self, unint=10, **kwargs):
            super(MyDenseModel, self).__init__(**kwargs)
            self.hideen1 = keras.layers.Dense(unint)
            self.hideen2 = keras.layers.Dense(unint)
            self.out = keras.layers.Dense(1)

        def call(self, inputs, training=None, mask=None):
            hiden1 = self.hideen1(inputs)
            hiden2 = self.hideen2(hiden1)
            out = self.out(hiden2)
            return out


    def launch():
        model = MyDenseModel()
        model.build(input_shape=(None, n_dim * time_step))
        model.summary()
        model.compile(loss="mse", optimizer=keras.optimizers.Nadam(learning_rate=0.001, decay=0.01),
                      metrics=keras.metrics.mean_absolute_percentage_error)
        history = model.fit(x_train_all, y_train_all, epochs=500, validation_data=(x_test_all, y_test_all))

        pre_now = model.predict(x_test_all)
        print("now_model_pre:", np.mean(keras.metrics.mean_absolute_percentage_error(y_test_all, pre_now)))


    # %%
    xg_reg = xgb.XGBRegressor(
        objective='reg:squarederror',
        colsample_bytree=0.3,
        learning_rate=0.01,
        max_depth=15,
        n_estimators=500,
        alpha=10
    )


    def display_scores(scores):
        print("Scores: ", scores)
        print("Mean: ", scores.mean())


    data_matrix = xgb.DMatrix(x_train_all, y_train_all)
    xg_reg.fit(x_train_all, y_train_all, eval_set=[(x_test_all, y_test_all)])
    scores = cross_val_score(xg_reg, x_train_all, y_train_all, scoring='neg_mean_absolute_percentage_error', cv=2)
    display_scores(scores)
    pred = xg_reg.predict(x_test_all)
    print(mean_absolute_percentage_error(y_test_all, pred))


    # %%
    # def gridsearch_cv(model, test_param, X, y, cv=5):
    #     gsearch = GridSearchCV(estimator=model, param_grid=test_param, scoring='neg_mean_absolute_percentage_error',
    #                            n_jobs=4, cv=cv)
    #     gsearch.fit(X, y)
    #     print('CV Results: ', gsearch.cv_results_)
    #     print('Best Params: ', gsearch.best_params_)
    #     print('Best Score: ', gsearch.best_score_)
    #     return gsearch.best_params_
    #
    #
    # param_test = {
    #     'max_depth': range(3, 10, 2),
    #     'min_child_weight': range(1, 10, 2),
    #     'n_estimators': [50, 100, 200, 300, 400],
    #     'learning_rate': [0.01, 0.05, 0.1]
    #
    # }
    # gridsearch_cv(xg_reg, param_test, x_train_all, y_train_all)
