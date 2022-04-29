# author:c19h
# datetime:2022/3/7 14:17
# %%
import os, sys

sys.path.append('./bin/utils')
import pandas as pd
import numpy as np
from cstemploader import CsTempLoader
from csvdataloader import CSVDataLoader
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import tensorflow.keras as keras
import tensorflow as tf
from collections import Counter


# %%
class Ad_Zl_Hour():
    def __init__(self):
        pass

    def load(self):
        self.PDData = CSVDataLoader('./data/tyc_127.csv')
        self.PDData.load(force_reload=True)
        self.PDTemp = CsTempLoader()
        self.PDTemp.load()
        self.data = self.PDData.HourData.join(ad.PDTemp.TempData)
        # './data/tyc_2987.csv'


def plot(all):
    fig = plt.figure(figsize=(24, 12))  # 调整画图空间的大小
    ax1 = fig.add_subplot(111)
    ax1.plot(all.index, all["用电量"], linestyle='-', marker='', markersize=10, c='r', label='val')  # 作图
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(all.index, all["温度"], linestyle='-', marker='', markersize=10, c='k', label='temp')  # 作图
    ax2.legend(loc='upper center')
    ax = plt.gca()
    date_format = mpl.dates.DateFormatter('%Y-%m-%d')  # 设定显示的格式形式
    ax.xaxis.set_major_formatter(date_format)  # 设定x轴主要格式
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(20))  # 设定坐标轴的显示的刻度间隔
    fig.autofmt_xdate()  # 防止x轴上的数据重叠，自动调整。
    plt.show()


def cal_similarity_day(a, b):
    a = np.array(a) + 10
    b = np.array(b) + 10
    delta1 = np.maximum(np.abs((a + b) / 2 - a), 0.3)
    delta2 = np.maximum(np.abs(np.sqrt(a * b) - a), 0.3)
    r1 = (np.exp(0.1 / delta1) + np.exp(0.1 / delta2)) / 2
    delta3 = np.maximum(np.abs(np.diff(b) - np.diff(a)), 0.3)
    r2 = np.exp(0.1 / delta3)
    r_mean = 0.5 * np.mean(r1) + 0.5 * np.mean(r2)
    return r_mean


if __name__ == "__main__":
    ad = Ad_Zl_Hour()
    ad.load()
    data = ad.data
    print(Counter(data['is_workday'] == 1))
    data.loc[((data.日用电量 < 200) & (data.is_workday == 1)), 'is_workday'] = 0
    print(Counter(data['is_workday'] == 1))
    workday = data[data['is_workday'] == 1]

    gb = workday.groupby(workday.index.date)
    temp = np.array([x[1]['温度'] for x in list(gb)])
    ind_similarity = []
    window = 6
    for i in range(0, len(temp) - window):
        a = np.apply_along_axis(cal_similarity_day, 1, temp[i:i + window], (temp[i + window],))
        ind_similarity.append(i + np.argmax(a))
    # %%
    val = np.array([x[1]['用电量'] for x in list(gb)])
    mean_temp = np.array([x[1]['平均温度'] for x in list(gb)])[:, 0]
    # %%
    x = np.empty((len(ind_similarity), 25))
    y = np.empty((len(ind_similarity), 24))
    for i in range(0, len(ind_similarity)):
        y[i, :] = val[i + window, :]
        x[i, :] = np.append(val[ind_similarity[i], :], mean_temp[i + window])
    # %%
    x_train_f, x_test, y_train_f, y_test = train_test_split(x, y, test_size=0.1, shuffle=True)
    # x_train, x_vali, y_train, y_vali = train_test_split(x_train_f, y_train_f, test_size=0.1, shuffle=True)

    # %%
    xg_reg = xgb.XGBRegressor(
        objective='reg:squarederror',
        colsample_bytree=0.3,
        learning_rate=0.01,
        max_depth=15,
        n_estimators=500,
        alpha=10
    )


    def mape(y_true, y_pred):
        return np.mean(np.abs(y_pred - y_true) / np.abs(y_true)) * 100


    def display_scores(scores):
        print("Scores: ", scores)
        print("Mean: ", scores.mean())


    # multioutputregressor = MultiOutputRegressor(xg_reg).fit(x_train_f, y_train_f)
    # # xg_reg.fit(x_train, y_train, eval_set=[(x_vali, y_vali)])
    # # scores = cross_val_score(multioutputregressor, x_train, y_train, scoring='neg_mean_absolute_percentage_error', cv=5)
    # # display_scores(scores)
    # pred = multioutputregressor.predict(x_test)
    # print(mean_absolute_percentage_error(y_test, pred))

    # %%
    param = {
        'max_depth': range(10, 20, 2),
        'min_child_weight': range(1, 10, 2),
        'n_estimators': [300, 400, 500],
        'learning_rate': [0.01, 0.05, 0.1]
    }
    gsc = GridSearchCV(estimator=xgb.XGBRegressor(), param_grid=param, cv=3, scoring='neg_mean_squared_error',
                       verbose=0, n_jobs=-1)

    grid_result = MultiOutputRegressor(gsc).fit(x_train_f, y_train_f)
    print('CV Results: ', grid_result.cv_results_)
    print('Best Params: ', grid_result.best_params_)
    print('Best Score: ', grid_result.best_score_)
