#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author:huxiao
# 加载从数据库导出的原始温度csv文件，并去除异常数据
# ==============================================================================

import numpy as np
import csv, sys, os
import matplotlib.pyplot as plt
from chinese_calendar import is_workday
import pandas as pd
import matplotlib as mpl
from scipy.special import erfinv
from sklearn.preprocessing import StandardScaler


class CSVTempLoader():
    def __init__(self):
        self.Files = []
        self.Files.append('./data/tyc_6079.csv')
        self.Files.append('./data/tyc_6081.csv')
        self.Files.append('./data/tyc_6083.csv')
        self.Files.append('./data/tyc_6085.csv')
        self.Files.append('./data/tyc_6087.csv')
        self.Files.append('./data/tyc_6089.csv')
        self.Files.append('./data/tyc_6091.csv')
        self.Files.append('./data/tyc_6093.csv')
        self.TempFileName = './tmp/temperature.csv'

    def _generate_day_data(self, pddata):
        fulldataday = pddata.resample('d').sum()
        fulldataday.drop(fulldataday[fulldataday['hour'] != 276].index, inplace=True)
        fulldatadaylist = [x.strftime('%Y-%m-%d') for x in fulldataday.index]
        pddata['tag_remove'] = pddata.index.map(lambda x: False if x.strftime('%Y-%m-%d') in fulldatadaylist else True)
        pddata.drop(pddata[pddata['tag_remove'] == True].index, inplace=True)
        pddata.drop(["tag_remove"], axis=1, inplace=True)
        # 关注工作日整天数据少的
        pddata['日用电量'] = pddata.index.map(lambda x: fulldataday.loc[x.strftime('%Y-%m-%d')]['用电量'])
        pddata['日用电比'] = pddata['用电量'] / pddata['日用电量']
        return pddata, fulldataday

    def _generate_hour_data(self, data):
        # 将原始数据处理为小时数据
        power = data.resample("1H").max()
        power['用电量'] = power['val'].shift(-1) - power['val']
        power['时间差'] = power['time2'].shift(-1) - power['time2']
        power.dropna(inplace=True)
        tdiff = pd.Timedelta('0 days 1 hours 10 minutes')
        tdiff2 = pd.Timedelta('0 days 0 hours 50 minutes')
        power = power.drop(power[power['时间差'] > tdiff].index)
        power = power.drop(power[power['时间差'] < tdiff2].index)
        power['is_workday'] = power.index.map(lambda x: 1 if True == is_workday(x) else 0)
        power["month"] = power.index.month
        power["hour"] = power.index.hour
        return power

    def _load_from_csv(self):
        main = None
        for i, filename in enumerate(self.Files):
            data = pd.read_csv(filename, infer_datetime_format=True)
            data.drop(data[data.q != 0].index, axis=0, inplace=True)
            data["time"] = pd.to_datetime(data["datatime"], infer_datetime_format=True)
            # data.rename(columns={'val':'val%d' % i}, inplace = True)
            data.set_index("time", inplace=True)
            data.drop(data[data['q'] != 0].index, inplace=True)
            data.drop(["datatime", "_msec", "q"], axis=1, inplace=True)

            dmean = data.resample("d").mean()
            dmean.rename(columns={'val': 'mean%d' % i}, inplace=True)
            if main is None:
                main = dmean
            else:
                main = pd.merge(main, dmean, on='time', how='outer')
            print('===', main.shape, dmean.shape)
        print(main.tail(3))
        main['mean'] = main.mean(axis=1)
        main.dropna(axis=0, subset=['mean'], inplace=True)
        main.dropna(axis=1, inplace=True)
        self._save_dump(main)

    def _save_dump(self, main):
        main.to_csv(self.TempFileName)
        self.TempData = main

    def _load_dump(self):
        self.TempData = pd.read_csv(self.TempFileName)
        self.TempData['time'] = pd.to_datetime(self.TempData['time'])
        self.TempData.set_index("time", inplace=True)

    def load(self, force_reload=False):
        if not os.path.exists(self.TempFileName) or \
                force_reload == True:
            self._load_from_csv()
        else:
            self._load_dump()


if __name__ == "__main__":
    cdl = CSVTempLoader()
    cdl.load(force_reload=True)
    print('complete!')


    # fig = plt.figure(figsize=(24, 12))  # 调整画图空间的大小
    # plt.plot(cdl.TempData.index, cdl.TempData["mean"], linestyle='', marker='.', markersize=10, c='r')  # 作图
    # ax = plt.gca()
    # date_format = mpl.dates.DateFormatter('%Y-%m-%d')  # 设定显示的格式形式
    # ax.xaxis.set_major_formatter(date_format)  # 设定x轴主要格式
    # ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(20))  # 设定坐标轴的显示的刻度间隔
    # fig.autofmt_xdate()  # 防止x轴上的数据重叠，自动调整。
    # plt.show()
    def scale_rankgauss(x, epsilon=1e-6):
        '''rankgauss'''
        x = x.argsort().argsort()  # rank
        x = (x / x.max() - 0.5) * 2  # scale
        x = np.clip(x, -1 + epsilon, 1 - epsilon)
        x = erfinv(x)
        return x


    scla = StandardScaler()
    # %%
    cdl.TempData['sclae'] = scale_rankgauss(cdl.TempData['mean'])
    cdl.TempData['sclae2'] = scla.fit_transform(np.array(cdl.TempData['mean']).reshape(-1, 1))

    # %%
    plt.hist(cdl.TempData['sclae2'], bins=50)
    plt.show()
