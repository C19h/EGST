# author:c19h
# datetime:2022/4/18 15:16
import configparser

import numpy as np
import sqlalchemy
import os
import pandas as pd


class DataLoader:
    def __init__(self, cfg):
        if not os.path.exists(cfg):
            print('未查询到配置文件，将新建配置文件保存至目录：{}'.format(cfg))
            config_file = cfg  # input("未查询到配置文件，请输入配置文件存放路径：")
            database_name = input("请输入数据库名称：")
            host = input("请输入数据库地址：")
            port = input("请输入数据库端口：")
            user = input("请输入数据库用户名称：")
            passwd = input("请输入数据库用户密码：")
            conf = configparser.ConfigParser()
            with open(config_file, 'w') as f:  # 写配置文件
                conf.add_section("Section")  # 在配置文件中增加一个段
                # 第一个参数是段名，第二个参数是选项名，第三个参数是选项对应的值
                conf.set("Section", "database_name", database_name)  # 要解析的数据库数据库
                conf.set("Section", "host", host)  # ip地址
                conf.set("Section", "port", port)  # 端口
                conf.set("Section", "user", user)  # 用户
                conf.set("Section", "passwd", passwd)  # 密码
                # 将conf对象中的数据写入到文件中
                conf.write(f)
        else:
            config_raw = configparser.RawConfigParser()
            config_raw.read(cfg, encoding="gbk")
            database_name = config_raw.get("Section", "database_name")
            host = config_raw.get("Section", "host")
            port = int(config_raw.get("Section", "port"))
            user = config_raw.get("Section", "user")
            passwd = config_raw.get("Section", "passwd")
        self.__engine = sqlalchemy.create_engine(
            'mysql+pymysql://{}:{}@{}:{}/{}'.format(user, passwd, host, port, database_name))
        self.data_sets = {'first_cluster_cell_voltage': ['tyc_{}'.format(id) for id in range(26777, 27017)],
                          'first_cluster_cell_temperature': ['tyc_{}'.format(id) for id in range(27017, 27097)],
                          'second_cluster_cell_voltage': ['tyc_{}'.format(id) for id in range(28789, 29029)],
                          'second_cluster_cell_temperature': ['tyc_{}'.format(id) for id in range(29029, 29109)],
                          'third_cluster_cell_voltage': ['tyc_{}'.format(id) for id in range(29290, 29530)],
                          'third_cluster_cell_temperature': ['tyc_{}'.format(id) for id in range(29530, 29610)],
                          'each_cluster_total_voltage': ['tyc_26756', 'tyc_28768', 'tyc_29269'],
                          'each_cluster_total_loop_current': ['tyc_26757', 'tyc_28769', 'tyc_29270'],
                          'each_cluster_soc': ['tyc_26759', 'tyc_28771', 'tyc_29272'],
                          'each_cluster_soh': ['tyc_26760', 'tyc_28772', 'tyc_29273']}

    def load_data(self, data_name, start=None, end=None):
        print('load data...')
        self.datas = []
        min_len = float('inf')
        flag = None
        for table in self.data_sets[data_name][start:end]:
            sql = "select * from {}".format(table)
            data = pd.read_sql(sql, self.__engine)
            if data_name.split('_')[-1] == 'voltage':
                data['val'] = data['val'] / 1000
            data.drop(data[data.q != 0].index, axis=0, inplace=True)
            if min_len > len(data):
                min_len = len(data)
                flag = data
            self.datas.append(data)
        flag = flag.rename(columns={'val': 'val_x'})
        for index, data in enumerate(self.datas):
            data.rename(columns={'val': 'val_y'}, inplace=True)
            new_df = pd.merge(flag, data, how='left', on='datatime')
            self.datas[index] = list(new_df['val_y'])
        self.datas = pd.DataFrame(self.datas)
        self.datas.dropna(axis=1, inplace=True)


if __name__ == '__main__':
    print(os.getcwd())
    dataloader = DataLoader('./config.ini')
    print(dataloader.data_sets.keys())
    dataloader.load_data('first_cluster_cell_voltage')
