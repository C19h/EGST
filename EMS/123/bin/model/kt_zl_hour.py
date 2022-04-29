#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author:huxiao
# 国际技术交流中心照明用电
# ==============================================================================
import sys, os, math
sys.path.append('./bin/utils')
import numpy as np
from scipy import optimize
from scipy.optimize import minimize
from configmanager import ConfigManager
from csvdataloader import CSVDataLoader
from csvtemploader import CSVTempLoader
from nightbasic import NightBasic
from hnxgboost import HNXGBoost
import pandas
from tools import *
class ZM_Jlzx_Hour():
    def __init__(self):
        self.SPLITRATE = 0.8

    def load(self):
        self.PDData = CSVDataLoader('./data/tyc_2987.csv')
        self.PDData.load()
        self.PDTemp = CSVTempLoader()
        self.PDTemp.load()
        # 根据日用电找到非工作日，从小时用电中去除
        pddata = self.PDData.DayData
        specialwork = pddata[pddata['用电量'] <= 220].index
        df = self.PDData.HourData
        indexes = None
        for r in specialwork:
            r2 = r + pandas.Timedelta("1 days")
            if indexes is None:
                indexes = (df.index >= r) & (df.index < r2)
            else:
                indexes += (df.index >= r) & (df.index < r2)
        self.PDData.HourData['is_workday'][indexes] = 0

    def _get_data_night(self, worktype):
        pddata = self.PDData.HourData
        pddata = pddata.join(self.PDTemp.HourData, on='time')
        if worktype == 1:
            pddata.drop(pddata[pddata['is_workday'] == 0].index, inplace = True)
            pddata.dropna(inplace = True)
        elif worktype == 2:
            pddata.drop(pddata[pddata['is_workday'] != 0].index, inplace = True)
            pddata.dropna(inplace = True)
        starttime = pddata.index[0]
        pddata['delta'] = pddata.index - starttime
        index1 = pddata['hour'] >= 22
        index2 = pddata['hour'] <= 5
        a = index1 + index2
        tmpdata = pddata[a]
        seq0 = np.array(tmpdata['delta'].astype('timedelta64[h]').astype(float)) / 24
        seq1 = np.array(tmpdata['month'])
        seq2 = np.array(tmpdata['用电量'])
        plotnp([seq2])
        return seq0, seq1, seq2

    def _get_data_by_hour(self, pddata, hour, removebelow = -1, removeabove = -1):
        if removeabove > 0:
            pddata.drop(pddata[pddata['用电量'] > removeabove].index, inplace = True)
            pddata.dropna(inplace = True)
        if removebelow > 0:
            pddata.drop(pddata[pddata['用电量'] < removebelow].index, inplace = True)
            pddata.dropna(inplace = True)
        tmpdata = pddata[pddata['hour'] == hour]
        seq0 = np.array(tmpdata['delta'].astype('timedelta64[D]').astype(float))
        seq2 = np.array(tmpdata['用电量'])
        seq1 = np.array(tmpdata['month']) / 24
        seq3 = np.array(tmpdata['max'])
        tmpdata2 = pddata[pddata['hour'] == hour+1]
        seq4 = np.array(tmpdata2['用电量'])

        tmpdata3 = pddata[pddata['hour'] == hour+2]
        seq5 = np.array(tmpdata3['用电量'])
        plotnp([seq2, seq4, seq5], ['%d~%d'%(hour, hour+1), '%d~%d'%(hour+1, hour+2), '%d~%d'%(hour+2, hour+3)])
        return seq0, seq2, seq1, seq3

    def model_basic_power(self):
        #通过夜间数据计算基础运行损耗
        x, y, = self._get_data_by_hour(0,0)
        self.nightmodel = NightBasic(x,y,self.SPLITRATE)
        self.nightmodel.fit()

    def _split_workday(self):
        pddata = self.PDData.HourData
        pddata = pddata.join(self.PDTemp.HourData, on='time')
        starttime = pddata.index[0]
        pddata.loc[:,'delta'] = pddata.index - starttime
        pdholiday = pddata[pddata['is_workday'] == 0]
        pdworkday = pddata[pddata['is_workday'] != 0]
        return pddata, pdworkday, pdholiday

    def model_total(self):
        pdall, pdworkday, pdholiday = self._split_workday()
        self._get_data_by_hour(pdworkday, 18)
        for i in range(24):
            x, y, month, temp = self._get_data_by_hour(pdworkday, i)
            nightmodel = NightBasic(x,y,self.SPLITRATE)
            loss = nightmodel.fit()
            print('workday hour %02d: %.1f' % (i, loss*100))
        listw = []
        for i in range(24):
            x, y, month, temp = self._get_data_by_hour(pdholiday, i)
            nightmodel = NightBasic(x,y,self.SPLITRATE)
            loss = nightmodel.fit()
            print('holiday hour %02d: %.1f' % (i, loss*100))
            listw.append(loss)
        print(np.mean(np.array(listw)))

    def model_xgboost(self):
        pdall, pdworkday, pdholiday = self._split_workday()
        self._get_data_by_hour(pdworkday, 18)
        for i in range(24):
            x, y, month, temp = self._get_data_by_hour(pdworkday, i)
            nightmodel = HNXGBoost(x,y,month,temp,self.SPLITRATE)
            loss = nightmodel.fit()
            print('workday hour %02d: %.1f' % (i, loss*100))
        listw = []
        for i in range(24):
            x, y, month, temp = self._get_data_by_hour(pdholiday, i)
            nightmodel = HNXGBoost(x,y,month,temp,self.SPLITRATE)
            loss = nightmodel.fit()
            print('holiday hour %02d: %.1f' % (i, loss*100))
            listw.append(loss)
        print(np.mean(np.array(listw)))

    def model_single(self):
        pdall, pdworkday, pdholiday = self._split_workday()
        i = 19
        x, y = self._get_data_by_hour(pdworkday, i)
        nightmodel = NightBasic(x,y,self.SPLITRATE)
        loss = nightmodel.fit()
        print('holiday hour %02d: %.1f' % (i, loss*100))



if __name__ == "__main__":
    zj = ZM_Jlzx_Hour()
    zj.load()
    zj.model_xgboost()
