# author:c19h
# datetime:2022/3/8 09:28
import pandas as pd
import numpy as np
import os, sys
from chinese_calendar import is_workday


# %%
class Calculate_Mape():
    def __init__(self):
        self.Files = []
        self.Files.append('./data/tyc_647.csv')  # 厂房照明
        self.Files.append('./data/tyc_907.csv')  # 厂房照明
        self.Files.append('./data/tyc_1167.csv')  # 厂房照明
        self.Files.append('./data/tyc_1687.csv')  # 宿舍照明
        self.Files.append('./data/tyc_1947.csv')  # 临建房照明
        self.Files.append('./data/tyc_1427.csv')  # 门卫照明
        self.Files.append('./data/tyc_387.csv')  # 后勤楼照明
        self.Files.append('./data/tyc_2207.csv')  # 厂房动力
        self.Files.append('./data/tyc_2467.csv')  # 厂房动力
        self.Files.append('./data/tyc_2727.csv')  # 厂房动力
        self.Files.append('./data/tyc_3279.csv')  # 新楼空调水泵
        self.Files.append('./data/tyc_2987.csv')  # 主楼空调

    def generate_data_hour(self, file):
        data = pd.read_csv(file)
        data.drop(data[data.q != 0].index, axis=0, inplace=True)
        data["time"] = pd.to_datetime(data["datatime"], infer_datetime_format=True)
        data["time2"] = pd.to_datetime(data["datatime"], infer_datetime_format=True)
        data.set_index("time", inplace=True)
        data.drop(["datatime", "_msec", "q"], axis=1, inplace=True)
        power = data.resample("1H").max()
        power['用电量'] = power['val'].shift(-1) - power['val']
        power['时间差'] = power['time2'].shift(-1) - power['time2']
        power.dropna(inplace=True)
        tdiff = pd.Timedelta('0 days 1 hours 10 minutes')
        tdiff2 = pd.Timedelta('0 days 0 hours 50 minutes')
        power = power.drop(power[power['时间差'] > tdiff].index)
        power = power.drop(power[power['时间差'] < tdiff2].index)
        power['is_workday'] = power.index.map(lambda x: 1 if True == is_workday(x) else 0)
        power["hour"] = power.index.hour
        power['hour'] += 1
        power.drop(power[power['用电量'] < 0.1].index, inplace=True)
        fulldataday = power.resample('d').sum()
        fulldataday.drop(fulldataday[fulldataday['hour'] != 300].index, inplace=True)
        # day = pd.Timedelta('1d')
        # fulldataday['time2'] = fulldataday.index
        # dt = fulldataday['time2']
        # in_block = ((dt - dt.shift(-1)).abs() == day) | (dt.diff() == day)
        # in_block1 = ((dt - dt.shift(-1)).abs() == day)
        # in_block2 = (dt.diff() == day)
        # filt = fulldataday.loc[in_block]
        # breaks = dt.diff() != day
        # groups = breaks.cumsum()
        # for _, frame in filt.groupby(groups):
        #     print(frame, end='\n\n')
        fulldatadaylist = [x.strftime('%Y-%m-%d') for x in fulldataday.index]
        power['tag_remove'] = power.index.map(lambda x: False if x.strftime('%Y-%m-%d') in fulldatadaylist else True)
        power.drop(power[power['tag_remove'] == True].index, inplace=True)
        power.drop(["tag_remove"], axis=1, inplace=True)
        power['hour'] -= 1
        return power[power['is_workday'] == 1], power[power['is_workday'] == 0]

    def mape(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def helper(self, data, timestep, wt):
        mapes = []
        for hour in range(0, 24):
            tem = data[data['hour'] == hour]
            tem = np.array(tem['用电量'])
            if len(tem) <= timestep:
                mapes.append(None)
            else:
                x = []
                y = []
                for i in range(0, len(tem) - timestep):
                    x.append(np.average(tem[i:i + timestep], weights=wt))
                    y.append(tem[i + timestep])
                res = self.mape(y, x)
                mapes.append(res)
        return mapes

    def cal(self, timestep, wt):
        all_mape_work = []
        all_mape_holiday = []
        for i, file in enumerate(self.Files):
            data_work, data_holiday = self.generate_data_hour(file)
            mapes_work = self.helper(data_work, timestep, wt)
            mapes_holiday = self.helper(data_holiday, timestep, wt)
            all_mape_work.append(mapes_work)
            all_mape_holiday.append(mapes_holiday)
        return np.array(all_mape_work).reshape(len(self.Files), -1), np.array(all_mape_holiday).reshape(len(self.Files),
                                                                                                        -1)


if __name__ == '__main__':
    cm = Calculate_Mape()
    wt = np.array([0.3, 0.3, 0.4])
    result_work, result_holiday = cm.cal(3, wt)