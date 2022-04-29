# author:c19h
# datetime:2022/2/24 17:48
from scipy.optimize import minimize
from csvdataloader import CSVDataLoader
from csvtemploader import CSVTempLoader
import numpy as np
import pandas as pd
data = pd.read_csv('mini_data.csv')
data = np.array([data]).reshape(3,-1)
#%%
def fun(para):
    a = para[0]
    c = para[1]
    d = para[2]
    e = para[3]
    # data: time, temp, power
    ans = (np.log(a * data[0, :] + 1000) + c) * d * data[1, :] - data[2, :] + 800 * e
    ans = np.sum(np.sqrt(ans * ans))
    return ans if not np.isnan(ans) else 999999999


def predict(res_x, x1, x2):
    a = res_x[0]
    c = res_x[1]
    d = res_x[2]
    e = res_x[3]
    return (np.log(a * x1 + 1000) + c) * d * x2 + 800 * e


x0 = [1, -8.21, -10, 0.5]
res = minimize(fun, x0)
pred_x = predict(res.x, data[0,:], data[1,:])
loss = np.average(np.abs(pred_x - data[2,:]) / data[2,:])
print('loss', loss)
# %%