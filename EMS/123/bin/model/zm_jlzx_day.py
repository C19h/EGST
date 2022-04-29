#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author:huxiao
# 国际技术交流中心照明用电
# ==============================================================================
import sys, os, math
sys.path.append('./bin/utils')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize
from csvdataloader import CSVDataLoader
from csvtemploader import CSVTempLoader
from torch.autograd import Variable
from tools import *
from rnn import *
from norm import Norm

class ZM_Jlzx_Day():
    def __init__(self):
        self.SPLITNUM = 400

    def load(self):
        self.PDData = CSVDataLoader('./data/tyc_127.csv')
        self.PDData.load()
        self.PDTemp = CSVTempLoader()
        self.PDTemp.load()

    def show_day_holiday(self):
        pddata = self.PDData.DayData
        pddata.drop(pddata[pddata['is_workday'] == 0].index, inplace = True)
        pddata = pddata.join(self.PDTemp.DayData, on='time')
        
        pddata.drop(pddata[pddata['用电量'] <= 220].index, inplace = True)
        pddata.drop(pddata[pddata['max'] <= 23].index, inplace = True)
        pddata.dropna(inplace = True)
        starttime = pddata.index[0]
        pddata['delta'] = pddata.index - starttime
        seq0 = np.array(pddata['delta'].astype('timedelta64[D]').astype(float))
        seq1 = np.array(pddata['max'])
        seq2 = np.array(pddata['用电量'])
        plotnp([seq2, seq1 * 10], ['power', 'temp-max'])
        return seq0, seq1, seq2

    def prepare_day_workday(self, pddata):
        pddata.drop(pddata[pddata['is_workday'] == 0].index, inplace = True)
        pddata = pddata.join(self.PDTemp.DayData, on='time')
        pddata.dropna(inplace = True)
        starttime = pddata.index[0]
        pddata['delta'] = pddata.index - starttime
        seq0 = np.array(pddata['delta'].astype('timedelta64[D]').astype(float))
        seq1 = np.array(pddata['max'])
        seq2 = np.array(pddata['用电量'])
        self._plot([seq1 * -1 * 30 + 800, seq2])
        return seq0, seq1, seq2

    def predict1(self, x1, x2):
        seq4 = self._fx2(x1, x2)
        self._plot([seq4 + seq2, seq2])
        return seq4

    def _fx2(self, x1, x2):
        a = self.x1[0]
        c = self.x1[1]
        d = self.x1[2]
        e = self.x1[3]
        # x: time, temp, power
        ans = (np.log(a*x1 + 1000) + c) * d * x2 + 800 * e
        return ans

    def _encode_x(self, x):
        return (x - 250) / (900 -250) 
    
    def _encode_y(self, y):
        return (y - 250) / (1100-250)

    def _decode_y(self, x):
        return x * (900 -250) + 250

    def _decode_y(self, y):
        return y * (1100 -250) + 250

    def train2(self, x, y):
        x = self._encode_x(x)
        y = self._encode_y(y)
        newx = y-x
        newx[1:] = newx[:-1]
        x = np.stack([x, newx], axis = 1)
        tx = torch.tensor(x, dtype=torch.float32)
        ty = torch.tensor(y[:, np.newaxis], dtype=torch.float32) 
        model = torch.nn.Sequential(
                  torch.nn.Linear(2, 32),
                  torch.nn.Tanh(),
                  torch.nn.Linear(32, 8),
                  torch.nn.Tanh(),
                  torch.nn.Linear(8, 4),
                  torch.nn.Tanh(),
                  torch.nn.Linear(4, 1)
            )
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        optimizer.zero_grad()
        for i in range(100):
            inputs = Variable(tx)
            targets = Variable(ty)
            pred = model(inputs)
            loss = loss_fn(pred,targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0: # 每1000次迭代打印一次误差值
                print('Epoch:{}, Loss:{:.5f}'.format(i+1, loss.item()))
        model.eval()
        self.model2 = model
    
    def predict2(self, x, y):
        x = self._encode_x(x)
        y = self._encode_y(y)
        # newx = np.zeros(x.shape[0])
        # newx[1:] = newx[:-1]
        # tx = np.stack([x, newx], axis = 1)
        # tx = torch.tensor(tx, dtype=torch.float32)
        # pred = self.model(tx)
        # npy = pred.detach().numpy()
        # newx = npy[:,0]-x
        newx = y - x
        newx[1:] = newx[:-1]
        tx = np.stack([x, newx], axis = 1)
        tx = torch.tensor(tx, dtype=torch.float32)
        pred = self.model2(tx)
        npy = pred.detach().numpy()
        y = self._decode_y(npy)

        return y

    def _fx(self, x1):
        a = x1[0]
        c = x1[1]
        d = x1[2]
        e = x1[3]
        # x: time, temp, power
        ans = (np.log(np.abs(a*self.x[0,:] + 1000)) + c) * d * self.x[1,:] - self.x[2,:] + 800 * e
        return ans

    def _fx_call(self, x1):
        res = self._fx(x1)
        ans = np.sum(np.sqrt(res * res))
        if not np.isnan(ans):
            return ans  
        else:
            return 9999999

    def solute_temp(self, time_arr, temp_arr, powerarr):
        self.x = np.array([time_arr, temp_arr, powerarr])
        x0 = [1, -8.21, -10, 0.5]
        res = minimize(self._fx_call, x0)
        print(res.success, res.fun, res.x)
        return res.x

    def get_pddata_workday(self):
        pdday = self.PDData.DayData.drop(self.PDData.DayData[self.PDData.DayData['is_workday'] == 24].index)
        pdday.drop(pdday[pdday['用电量'] <= 220].index, inplace = True)
        return pdday

    def train_function(self):
        #用拟合的方式与预测
        wdata = zj.get_pddata_workday()
        x0_1,x0_2, y = zj.prepare_day_workday(wdata)
        # 模型1
        train_x1 = x0_1[:self.SPLITNUM]
        train_x2 = x0_2[:self.SPLITNUM]
        train_y1 = y[:self.SPLITNUM]
        test_x1 = x0_1[self.SPLITNUM:]
        test_x2 = x0_2[self.SPLITNUM:]
        test_y1 = y[self.SPLITNUM:]
        self.x1 = self.solute_temp(train_x1,train_x2,train_y1)
        pred_x = zj.predict1(test_x1,test_x2)
        loss = np.average(np.abs(pred_x - test_y1) /test_y1)
        print('loss1', loss)
        zj._plot([pred_x, test_y1])

    def train_rnn(self):
        #使用rnn预测
        wdata = zj.get_pddata_workday()
        x0_1,x0_2, y = zj.prepare_day_workday(wdata)
        # 模型1
        train_x1 = x0_1[:self.SPLITNUM]
        train_x2 = x0_2[:self.SPLITNUM]
        train_y1 = y[:self.SPLITNUM]
        test_x1 = x0_1[self.SPLITNUM:]
        test_x2 = x0_2[self.SPLITNUM:]
        test_y1 = y[self.SPLITNUM:]

        org_x = np.stack([x0_1, x0_2], axis = 1)
        org_y = y[:, np.newaxis]
        norm_x = Norm(org_x)
        norm_y = Norm(org_y)
        std_x = norm_x.encode(org_x)
        std_y = norm_y.encode(org_y)
        train_x = std_x[:self.SPLITNUM]
        train_y = std_y[:self.SPLITNUM]
        test_x = std_x[self.SPLITNUM:]
        test_y = std_y[self.SPLITNUM:]
        dstrain = DataSet(train_x, train_y)
        dstest = DataSet(test_x, test_y)
        tmp = norm_y.decode(dstest[0][1])
        model = Model()
        model.train(dstrain, dstest, norm_y)
        print('==== train over ====')
        
        x = None
        for i in range(self.SPLITNUM + 5, org_x.shape[0]):
            tmp = np.array(std_x[i - 6:i], dtype=np.float32)
            if x is None:
                x = tmp[np.newaxis, :,:]
            else:
                x = np.vstack((x, tmp[np.newaxis, :,:]))
        predy = model.predict(x)
        y = predy[:,-1,:]
        predy2 = norm_y.decode(y)
        test_y = norm_y.decode(std_y)[self.SPLITNUM+5:,0]
        pred_y = predy2[:,0]

        loss_arr = pred_y - test_y
        loss = np.average(np.abs(loss_arr) / test_y)
        print('loss2', loss)
        zj._plot([pred_y, test_y ])

if __name__ == "__main__":
    zj = ZM_Jlzx_Day()
    zj.load()
    zj.show_day_holiday()
    zj.train_rnn()
    
