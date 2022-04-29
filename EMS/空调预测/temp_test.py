# author:c19h
# datetime:2022/2/23 08:38
import pandas as pd
from chinese_calendar import is_workday
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# %%
def process(path):
    en_te = pd.read_csv(path)
    en_te["time"] = pd.to_datetime(en_te["datatime"], infer_datetime_format=True)
    en_te["time2"] = pd.to_datetime(en_te["datatime"], infer_datetime_format=True)
    en_te.set_index("time", inplace=True)
    en_te.drop(en_te[en_te['q'] != 0].index, inplace=True)
    en_te.drop(["datatime", "_msec", "q"], axis=1, inplace=True)
    max_te = en_te.resample("d").mean()
    max_te.dropna(inplace=True)
    max_te['is_workday'] = max_te.index.map(lambda x: is_workday(x))
    # max_te.drop(max_te[max_te['is_workday'] == False].index, inplace=True)
    max_te["month"] = max_te.index.month
    max_te["day"] = max_te.index.day
    return max_te


fac1_tem_1f = process('./data/tyc_6079.csv')
fac1_tem_2f = process('./data/tyc_6081.csv')
fac2_tem_1f = process('./data/tyc_6083.csv')
fac2_tem_2f = process('./data/tyc_6085.csv')
fac3_tem_1f = process('./data/tyc_6087.csv')
fac4_tem_1f = process('./data/tyc_6089.csv')
fac5_tem_1f = process('./data/tyc_6091.csv')
hq_tem_1f = process('./data/tyc_6093.csv')


# %%
def plot(inp):
    fig = plt.figure(figsize=(24, 12))  # 调整画图空间的大小
    plt.plot(inp.index, inp["val"], linestyle='', marker='.', markersize=10, c='r')  # 作图
    ax = plt.gca()
    date_format = mpl.dates.DateFormatter('%Y-%m-%d')  # 设定显示的格式形式
    ax.xaxis.set_major_formatter(date_format)  # 设定x轴主要格式
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(30))  # 设定坐标轴的显示的刻度间隔
    fig.autofmt_xdate()  # 防止x轴上的数据重叠，自动调整。
    plt.show()


# %%
plot(fac1_tem_1f)
plot(fac4_tem_1f)
plot(fac5_tem_1f)
