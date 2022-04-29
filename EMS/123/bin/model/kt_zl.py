#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author:huxiao
# 主楼空调用电
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

class KT_ZL():
    def __init__(self):
        pass

    def load(self):
        self.PDData = CSVDataLoader('./data/tyc_2987.csv')
        self.PDData.load()
        self.PDTemp = CSVTempLoader()
        self.PDTemp.load()

    def show_hour_data(self):
        pddata = self.PDData.HourData
        pddata = pddata.join(self.PDTemp.HourData, on='time')
        pddata.drop(pddata[pddata['is_workday'] == 0].index, inplace = True)
        pddata.drop(pddata[pddata['用电量'] < 20].index, inplace = True)
        #pddata.drop(pddata[pddata['max'] < 25].index, inplace = True)
        #seq1 = np.array(pddata[pddata['hour'] == 5]['用电量'])
        #seq2 = np.array(pddata[pddata['hour'] == 6]['用电量'])
        #seq3 = np.array(pddata[pddata['hour'] == 7]['用电量'])
        seq3 = np.array(pddata[pddata['hour'] == 14]['用电量'])
        seq4 = np.array(pddata[pddata['hour'] == 14]['max']) * 7
        row = pddata.iloc[100,:]
        self._plot([seq3,seq4])

    def show_day_data(self):
        pddata = self.PDData.DayData
        pddata = pddata.join(self.PDTemp.DayData, on='time')
        pddata.drop(pddata[pddata['is_workday'] == 0].index, inplace = True)
        pddata.drop(pddata[pddata['用电量'] < 250].index, inplace = True)
        seq1 = np.array(pddata['用电量'])
        seq2 = np.array(pddata['max']) * 100
        seq3 = np.array(pddata['month']) /24 * 100
        self._plot([seq1,seq2, seq3])

    def prepare_day_workday(self, pddata):
        pddata = pddata.join(self.PDTemp.HourData, on='time')
        pddata.drop(pddata[pddata['is_workday'] == 0].index, inplace = True)
        pddata.drop(pddata[pddata['用电量'] < 250].index, inplace = True)
        pddata.dropna(inplace = True)
        starttime = pddata.index[0]
        pddata['delta'] = pddata.index - starttime
        seq0 = np.array(pddata['delta'].astype('timedelta64[D]').astype(float))
        seq1 = np.array(pddata['max'])
        seq2 = np.array(pddata['用电量'])
        return seq0, seq1, seq2
    
    def train1(self, seq0,seq1,seq2):
        self.x1 = self.solute_temp(seq0,seq1,seq2)

    def predict1(self, x1, x2):
        seq4 = self._fx2(x1, x2)
        #self._plot([seq4 + seq2, seq2])
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
        ans = (np.log(a*self.x[0,:] + 1000) + c) * d * self.x[1,:] - self.x[2,:] + 800 * e
        return ans

    def _fx_call(self, x1):
        res = self._fx(x1)
        ans = np.sum(np.sqrt(res * res))
        return ans if not np.isnan(ans) else 999999999

    def solute_temp(self, time_arr, temp_arr, powerarr):
        self.x = np.array([time_arr, temp_arr, powerarr])
        x0 = [1, -8.21, -10, 0.5]
        res = minimize(self._fx_call, x0)
        print(res.fun, res.x)
        return res.x

    def get_pddata_workday(self):
        pdday = self.PDData.DayData.drop(self.PDData.DayData[self.PDData.DayData['is_workday'] != 24].index)
        pdday.drop(pdday[pdday['用电量'] <= 220].index, inplace = True)
        return pdday

    def _plot(self, npdata):
        for i, r in enumerate(npdata):
            plt.plot(r, label='data%d' % (i+1))
        plt.legend()
        plt.show()

if __name__ == "__main__":
    zj = KT_ZL()
    zj.load()
    zj.show_hour_data()
    wdata = zj.get_pddata_workday()
    x0_1,x0_2, y = zj.prepare_day_workday(wdata)
    # 模型1
    SPLITNUM = 250
    train_x1 = x0_1[:SPLITNUM]
    train_x2 = x0_2[:SPLITNUM]
    train_y1 = y[:SPLITNUM]
    test_x1 = x0_1[SPLITNUM:]
    test_x2 = x0_2[SPLITNUM:]
    test_y1 = y[SPLITNUM:]
    zj.train1(train_x1,train_x2,train_y1)
    pred_x = zj.predict1(test_x1,test_x2)
    loss = np.average(np.abs(pred_x - test_y1) /test_y1)
    print('loss1', loss)
    zj._plot([pred_x, test_y1])
    # 模型2
    train_x = zj.predict1(train_x1,train_x2)
    train_y = train_y1
    test_x = zj.predict1(test_x1,test_x2)
    test_y = test_y1

    zj.train2(train_x,train_y)
    pred_y = zj.predict2(test_x, test_y)

    loss_arr = pred_y - test_y
    loss = np.average(np.abs(loss_arr) / test_y)
    print('loss2', loss)
    zj._plot([pred_y, test_y])
