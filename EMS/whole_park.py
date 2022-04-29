# author:c19h
# datetime:2022/3/8 16:36
import pandas as pd
import numpy as np
import os, sys
from chinese_calendar import is_workday

# %%

Files = []
Files.append('./data/tyc_647.csv')  # 厂房照明
Files.append('./data/tyc_907.csv')  # 厂房照明
Files.append('./data/tyc_1167.csv')  # 厂房照明
Files.append('./data/tyc_1687.csv')  # 宿舍照明
Files.append('./data/tyc_1947.csv')  # 临建房照明
Files.append('./data/tyc_1427.csv')  # 门卫照明
Files.append('./data/tyc_387.csv')  # 后勤楼照明
Files.append('./data/tyc_2207.csv')  # 厂房动力
Files.append('./data/tyc_2467.csv')  # 厂房动力
Files.append('./data/tyc_2727.csv')  # 厂房动力
Files.append('./data/tyc_3279.csv')  # 新楼空调水泵
Files.append('./data/tyc_2987.csv')  # 主楼空调

park = []
for files in Files:
    data = pd.read_csv(files)
    data.drop(data[data.q != 0].index, axis=0, inplace=True)
    data["time"] = pd.to_datetime(data["datatime"], infer_datetime_format=True)
    data.set_index("time", inplace=True)
    data.drop(["datatime", "_msec", "q"], axis=1, inplace=True)
    power = data.resample("1H").max()
    park.append(power)
# %%
pd_park = pd.concat(park, join='inner', axis=1, )
pd_park["sum"] = pd_park.sum(axis=1)
pd_park['time'] = pd_park.index
pd_park = pd_park[["sum", 'time']]
# %%
pd_park['用电量'] = pd_park['sum'].shift(-1) - pd_park['sum']
pd_park['时间差'] = pd_park['time'].shift(-1) - pd_park['time']
pd_park.dropna(inplace=True)
tdiff = pd.Timedelta('0 days 1 hours 10 minutes')
tdiff2 = pd.Timedelta('0 days 0 hours 50 minutes')
pd_park = pd_park.drop(pd_park[pd_park['时间差'] > tdiff].index)
pd_park = pd_park.drop(pd_park[pd_park['时间差'] < tdiff2].index)
pd_park['is_workday'] = pd_park.index.map(lambda x: 1 if True == is_workday(x) else 0)
pd_park["hour"] = pd_park.index.hour
pd_park['hour'] += 1
# pd_park.drop(pd_park[pd_park['用电量'] < 0.1].index, inplace=True)
fulldataday = pd_park.resample('d').sum()
