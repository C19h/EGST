# author:c19h
# datetime:2022/3/7 14:00
# %%
import pandas as pd
import os
import matplotlib.pyplot as plt


# %%
class CsTempLoader():
    def __init__(self):
        self.TempFileName = './tmp/CStemperature.csv'

    def _save_dump(self, main):
        main.to_csv(self.TempFileName)
        self.TempData = main

    def _load_dump(self):
        self.TempData = pd.read_csv(self.TempFileName, index_col=0)
        self.TempData.index = pd.to_datetime(self.TempData.index)

    def _generate_data(self):
        data = pd.read_csv('./data/气象数据/576870-99999-2018.out', sep='\\s+', skiprows=[0], header=None)
        temp = pd.DataFrame(data[21].values, index=pd.to_datetime(data[2], format='%Y%m%d%H%M'), columns=['温度'])
        for i in range(19, 23):
            data = pd.read_csv('./data/气象数据/576870-99999-20{}.out'.format(i), sep='\\s+', skiprows=[0], header=None)
            tem = pd.DataFrame(data[21].values, index=pd.to_datetime(data[2], format='%Y%m%d%H%M'), columns=['温度'])
            temp = pd.concat([temp, tem])
        temp.drop(temp[temp['温度'] == '****'].index, inplace=True)
        temp = temp.resample('1h').fillna(None)
        data_interpolated = temp.astype(float).interpolate(method='spline', order=3)
        temp_interpolated = (data_interpolated - 32) / 1.8
        temp_interpolated['平均温度'] = temp_interpolated.index.map(lambda x: temp_interpolated.loc[x.strftime('%Y-%m-%d')]['温度'].mean())
        self._save_dump(temp_interpolated)

    def load(self, force_reload=False):
        if not os.path.exists(self.TempFileName) or \
                force_reload == True:
            self._generate_data()
        else:
            self._load_dump()


if __name__ == "__main__":
    loader = CsTempLoader()
    loader.load()
    # %%
    plt.plot(loader.TempData['温度'], linestyle='', marker='.', markersize=1, c='r')
