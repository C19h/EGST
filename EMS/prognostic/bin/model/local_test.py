# author:c19h
# datetime:2022/4/19 11:39
import sys

import numpy as np

from bin.utils.security_warning import SecurityWarning
from bin.utils.dataloader import DataLoader
from bin.utils.tools import *
import os
import pandas as pd


class Main:
    def __init__(self):
        pass

    def load(self, component, start=None, end=None):
        self.data = DataLoader('./bin/utils/config.ini')
        self.data.load_data(component, start, end)

    def sw(self, window, amplitude, time_step, threshold, tolerance, method='adjacent'):
        s = SecurityWarning(window, amplitude, time_step)
        if method == 'adjacent':
            res = s.correlation_analysis(self.data.datas.iloc[:, 10000:])
        else:
            res = s.correlation_analysis_mean(self.data.datas[:, 10000:])
        num_abnormal = []
        for r in res:
            index_tem = np.count_nonzero(r < threshold)
            num_abnormal.append(index_tem)
        num_abnormal = np.array(num_abnormal)
        index = np.where(num_abnormal > tolerance)

        return num_abnormal, index, np.array(res)


if __name__ == '__main__':
    print(os.getcwd())
    m = Main()
    m.load('first_cluster_cell_voltage')
    # %%
    # num_abnormal, index, res = m.sw(10, 0.01, 30, 0.95, 10)
    # %%
    # all = pd.DataFrame(m.data.datas.mean(axis=0)).T.append(m.data.datas.iloc[index])
    # %%
    # plotys(m.data.datas.iloc[np.append(index[0], [0, 1, 2, 3]), :])
