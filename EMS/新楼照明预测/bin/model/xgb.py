# author:c19h
# datetime:2022/3/4 08:47
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
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, cross_val_score
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

    def get_pddata_workday(self):
        # 天
        pdday = self.PDData.DayData.drop(self.PDData.DayData[self.PDData.DayData['is_workday'] != 24].index)
        pdday.drop(pdday[pdday['用电量'] <= 220].index, inplace=True)
        pdday.drop(pdday[pdday['is_workday'] == 0].index, inplace=True)
        pdday = pdday.join(self.PDTemp.DayData, on='time')
        pdday.dropna(inplace=True)
        starttime = pdday.index[0]
        pdday['delta'] = (pdday.index - starttime)
        seq0 = np.array(pdday['delta'].astype('timedelta64[D]').astype(float))
        pdday['delta'] = seq0
        # 小时
        daylist = [x.strftime('%Y-%m-%d') for x in pdday.index]
        hw_e = self.PDData.HourData.drop(self.PDData.HourData[self.PDData.HourData['is_workday'] != 1].index)
        hw_e = hw_e.dropna()
        hw_e['tag'] = hw_e.index.map(lambda x: True if x.strftime('%Y-%m-%d') in daylist else False)
        hw_e = hw_e.drop(hw_e[hw_e['tag'] == False].index)
        hw_et = hw_e.join(self.PDTemp.HourData, on='time')
        hw_et.dropna(inplace=True)
        return pdday, hw_et


if __name__ == "__main__":
    # %%
    zj = ZM_Jlzx()
    zj.load()
    zj.get_pddata_workday()
    # 模型1
    # x0_1 = x0_1.reshape(-1, 1)
    # x0_2 = x0_2.reshape(-1, 1)
    # X = np.concatenate([x0_1, x0_2], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(pddata[['max', 'delta', '用电量']], pddata['用电量'], test_size=0.2,
                                                        shuffle=False)
    scx = StandardScaler()
    scy = StandardScaler()
    X_train_s = scx.fit_transform(X_train)
    X_test_s = scx.transform(X_test)
    # %%

    time_step = 5
    n_dim = X_train.shape[1]
    x_train_all = np.zeros((len(X_train) - time_step, n_dim * time_step))
    y_train_all = np.zeros((len(y_train) - time_step, 1))
    x_test_all = np.zeros((len(X_test) - time_step, n_dim * time_step))
    y_test_all = np.zeros((len(y_test) - time_step, 1))
    for i in range(0, len(X_train_s) - time_step):
        x_train_all[i, :] = np.array(X_train_s[i:i + time_step, :]).reshape(-1)
        y_train_all[i, :] = np.array(y_train[i + time_step])

    for i in range(0, len(X_test_s) - time_step):
        x_test_all[i, :] = np.array(X_test_s[i:i + time_step, :]).reshape(-1)
        y_test_all[i, :] = np.array(y_test[i + time_step])

    xg_reg = xgb.XGBRegressor(
        objective='reg:squarederror',
        colsample_bytree=0.3,
        learning_rate=0.1,
        max_depth=10,
        n_estimators=200,
        alpha=10
    )


    def display_scores(scores):
        print("Scores: ", scores)
        print("Mean: ", scores.mean())


    data_matrix = xgb.DMatrix(x_train_all, y_train_all)
    xg_reg.fit(x_train_all, y_train_all, eval_set=[(x_test_all, y_test_all)],
               eval_metric='mape', early_stopping_rounds=50)
    scores = cross_val_score(xg_reg, x_train_all, y_train_all, scoring='neg_mean_absolute_percentage_error', cv=5)
    display_scores(scores)
    pred = xg_reg.predict(x_test_all)
    train = xg_reg.predict(x_train_all)
    print(mean_absolute_percentage_error(y_test_all, pred))
    print(mean_absolute_percentage_error(y_train_all, train))
    # %%
    plt.plot(y_test_all, 'r', label="true")
    plt.plot(pred, 'k', label="pre")
    plt.legend()
    plt.show()


    def get_mape_with_hour(pre):
        pre_last = pre
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
        return mean_absolute_percentage_error(np.array(a['每小时预测用电']),
                                              np.array(a['用电量'])), mean_absolute_percentage_error(
            np.array(a['用电量'][:-24]), np.array(a['用电量'][24:]))


    get_mape_with_hour(pred)
