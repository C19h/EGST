#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author:huxiao
# 夜间基础用电模型
# ==============================================================================
import sys, os, math
sys.path.append('./bin/utils')
import numpy as np
from tools import *
from  sklearn import datasets 
import pandas as pd 
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class HNXGBoost():
    def __init__(self, x, y, month, temp, splitrate):
        self.SPLITRATE = splitrate
        self.SPLITNUM = int(x.shape[0] * self.SPLITRATE)

        t1 = y[0:-3]
        t2 = y[1:-2]
        t3 = y[2:-1]
        t4 = x[3:]
        t5 = month[3:]
        t6 = temp[3:]
        self.x = np.stack([t1,t2,t3,t4,t5,t6], axis=1)

        self.y = y[3:]

    def fit(self):
        data_matrix = xgb.DMatrix(self.x,self.y)
        xg_reg= xgb.XGBRegressor(
            objective='reg:squarederror',
            colsample_bytree=0.3,
            learning_rate=0.1,
            max_depth=10,
            n_estimators=40,
            alpha=10
        )
        x_train,x_test,y_train  ,y_test = train_test_split(self.x,self.y,test_size=0.2)
        xg_reg.fit(x_train,y_train)
        pred = xg_reg.predict(x_test)
        loss = np.mean(np.abs(pred - y_test) / y_test)
        return loss

