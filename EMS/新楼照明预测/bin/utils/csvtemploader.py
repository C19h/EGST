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
        self.HourFileName = './tmp/temperature_hour.csv'
        self.DayFileName = './tmp/temperature_day.csv'

    def _generate_day_data(self, pddata):
        fmax = pddata.resample('d').max()
        fmax.drop(['min'], axis=1, inplace=True)
        fmin = pddata.resample('d').min()
        fmin.drop(['max'], axis=1, inplace=True)
        main = pd.merge(fmax, fmin, on='time', how='outer')
        return main

    def _generate_hour_data(self):
        # 将原始数据处理为小时数据
        main = None
        for i, filename in enumerate(self.Files):
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
            print('===', main.shape, data.shape)

        colsmax = ['max%d' % x for x in range(len(self.Files))]
        colsmin = ['min%d' % x for x in range(len(self.Files))]
        main['max'] = main[colsmax].mean(axis=1, skipna=True)
        main['min'] = main[colsmin].mean(axis=1, skipna=True)

        for i in range(len(self.Files)):
            main.drop(['max%d' % i, 'min%d' % i], axis=1, inplace=True)
        main.dropna(inplace=True)
        return main

    def _load_from_csv(self):
        pdhour = self._generate_hour_data()
        pdday = self._generate_day_data(pdhour)
        self._save_dump(pdhour, pdday)

    def _save_dump(self, pdhour, pdday):
        pdhour.to_csv(self.HourFileName)
        pdday.to_csv(self.DayFileName)
        self.HourData = pdhour
        self.DayData = pdday

    def _load_dump(self):
        self.HourData = pd.read_csv(self.HourFileName)
        self.HourData['time'] = pd.to_datetime(self.HourData['time'])
        self.HourData.set_index("time", inplace=True)
        self.DayData = pd.read_csv(self.HourFileName)
        self.DayData['time'] = pd.to_datetime(self.DayData['time'])
        self.DayData.set_index("time", inplace=True)

    def load(self, force_reload=False):
        if not os.path.exists(self.HourFileName) or \
                not os.path.exists(self.DayFileName) or \
                force_reload == True:
            self._load_from_csv()
        else:
            self._load_dump()


if __name__ == "__main__":
    cdl = CSVTempLoader()
    cdl.load(force_reload=True)
    print('complete!')
