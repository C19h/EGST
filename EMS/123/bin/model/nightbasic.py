#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author:huxiao
# 夜间基础用电模型
# ==============================================================================
import sys, os, math
sys.path.append('./bin/utils')
import numpy as np
from tools import *
from sklearn import linear_model
class NightBasic():
    def __init__(self, x, y, splitrate):
        self.SPLITRATE = splitrate
        self.SPLITNUM = int(x.shape[0] * self.SPLITRATE)
        self.x = x
        self.y = y

    def fit(self):
        return self.fit_lastday()

    def fit_poly(self):
        train_x = self.x[:self.SPLITNUM, np.newaxis]
        train_y = self.y[:self.SPLITNUM, np.newaxis]
        test_x = self.x[self.SPLITNUM:, np.newaxis]
        test_y = self.y[self.SPLITNUM:]
        self.model = linear_model.LinearRegression()
        self.model.fit(train_x, train_y)
        pred_y = model.predict(test_x)[:,0]
        loss = np.mean(np.abs(predy - test_y) / test_y)
        print('model_basic_power loss', loss)
        plotnp([test_y, pred_y])

    def fit_lastday(self, step=1):
        test_y = self.y[self.SPLITNUM:]
        ratelist = np.arange(step) + 1
        pred_y = self.y[self.SPLITNUM-1:-1].copy() * ratelist[step - 1] / np.sum(ratelist)
        for i in range(2, step + 1):
            pred_y += self.y[self.SPLITNUM - i: -1 * i] * ratelist[step - i] / np.sum(ratelist)
        loss = np.mean(np.abs(pred_y - test_y) / test_y)
        plotnp([test_y, pred_y])
        return loss

    def fit_lastweek(self, step=1):
        test_y = self.y[self.SPLITNUM:]
        ratelist = np.arange(step) + 1
        pred_y = self.y[self.SPLITNUM-7:-7].copy() * ratelist[step - 1] / np.sum(ratelist)
        for i in range(2, step + 1):
            pred_y += self.y[self.SPLITNUM - i * 7: -7 * i] * ratelist[step - i] / np.sum(ratelist)
        loss = np.mean(np.abs(pred_y - test_y) / test_y)
        plotnp([test_y, pred_y])
        return loss
