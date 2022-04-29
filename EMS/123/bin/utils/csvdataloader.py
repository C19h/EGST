#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author:huxiao
# 加载从数据库导出的原始csv文件，并去除异常数据
# ==============================================================================

import numpy as np
import csv, sys, os
import matplotlib.pyplot as plt
from chinese_calendar import is_workday
import pandas as pd

class CSVDataLoader():
    def __init__(self, csvfilename):
        self.FileName = csvfilename
        basename = self.FileName.split('/')[-1]
        self.HourFileName = "./tmp/%s_hour.csv" % basename[:-4]
        self.DayFileName = "./tmp/%s_day.csv" % basename[:-4]

    def _generate_day_data(self, pddata):
        fulldataday = pddata.resample('d').sum()
        fulldataday.drop(fulldataday[fulldataday['hour'] != 276].index, inplace = True)   
        fulldatadaylist = [x.strftime('%Y-%m-%d') for x in fulldataday.index]
        pddata['tag_remove'] = pddata.index.map(lambda x: False if x.strftime('%Y-%m-%d') in fulldatadaylist else True)
        pddata.drop(pddata[pddata['tag_remove'] == True].index, inplace = True)
        pddata.drop(["tag_remove"], axis=1, inplace=True)
        #关注工作日整天数据少的
        pddata['日用电量'] = pddata.index.map(lambda x: fulldataday.loc[x.strftime('%Y-%m-%d')]['用电量'])
        pddata['日用电比'] = pddata['用电量']/ pddata['日用电量']
        return pddata, fulldataday

    def _generate_hour_data(self, data):
        #将原始数据处理为小时数据
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
        data = pd.read_csv(self.FileName, infer_datetime_format=True)
        data.drop(data[data.q != 0].index, axis=0, inplace=True)
        data["time"] = pd.to_datetime(data["datatime"], infer_datetime_format=True)
        data["time2"] = pd.to_datetime(data["datatime"], infer_datetime_format=True)

        data.set_index("time", inplace=True)
        data.drop(data[data['q'] != 0].index, inplace = True)
        data.drop(["datatime", "_msec", "q"], axis=1, inplace=True)

        power = self._generate_hour_data(data)
        pdhour, pdday = self._generate_day_data(power)
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
        self.DayData = pd.read_csv(self.DayFileName)
        self.DayData['time'] = pd.to_datetime(self.DayData['time'])
        self.DayData.set_index("time", inplace=True)

    def load(self, force_reload = False):
        if not os.path.exists(self.DayFileName) or \
            not os.path.exists(self.HourFileName) or \
            force_reload == True:
            self._load_from_csv()
        else:
            self._load_dump()

if __name__ == "__main__":
    cdl = CSVDataLoader('./data/tyc_127.csv')
    cdl.load(force_reload=True)
    print('complete!')