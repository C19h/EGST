# author:c19h
# datetime:2022/4/11 16:46
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import sqlalchemy
import random
from sqlalchemy.ext.declarative import declarative_base

# %%

first_cluster_cell_voltage = ['tyc_{}'.format(id) for id in range(26777, 27017)]
first_cluster_cell_temperature = ['tyc_{}'.format(id) for id in range(27017, 27097)]
second_cluster_cell_voltage = ['tyc_{}'.format(id) for id in range(28789, 29029)]
second_cluster_cell_temperature = ['tyc_{}'.format(id) for id in range(29029, 29109)]
third_cluster_cell_voltage = ['tyc_{}'.format(id) for id in range(29290, 29530)]
third_cluster_cell_temperature = ['tyc_{}'.format(id) for id in range(29530, 29610)]
each_cluster_total_voltage = ['tyc_26756', 'tyc_28768', 'tyc_29269']
each_cluster_total_loop_current = ['tyc_26757', 'tyc_28769', 'tyc_29270']
each_cluster_soc = ['tyc_26759', 'tyc_28771', 'tyc_29272']
each_cluster_soh = ['tyc_26760', 'tyc_28772', 'tyc_29273']

voltage = first_cluster_cell_voltage + second_cluster_cell_voltage + third_cluster_cell_voltage + each_cluster_total_voltage


# %%
def get_data(tables_name):
    engine = sqlalchemy.create_engine(
        'mysql+pymysql://{}:{}@{}:{}/{}'.format('root', 'pwdzh951230', 'localhost', 3306, 'ems'))
    datas = []
    for table in tables_name:
        sql = "select * from {}".format(table)
        data = pd.read_sql(sql, engine)
        if table in voltage:
            data['val'] = data['val'] / 1000
        data.drop(data[data.q != 0].index, axis=0, inplace=True)
        datas.append(data)
    return datas


rand = np.random.RandomState(144)
T = 10
c_voltage = 0.1
c_current = 1
c_temperature = 1
time_step = 120
cor = np.append(c_current * np.ones(int(T / 2)), -c_current * np.ones(int(T / 2)))
correct = np.tile(cor, int(time_step / T))


def cal_person(y1, y2):
    person_list = []
    person_list_cor = []
    for i in range(min(len(y1), len(y2)) - time_step + 1):
        tem = scipy.stats.pearsonr(y1[i:i + time_step], y2[i:i + time_step])[0]
        tem_cor = scipy.stats.pearsonr(y1[i:i + time_step] + correct, y2[i:i + time_step] + correct)[0]
        person_list.append(tem)
        person_list_cor.append(tem_cor)
    return person_list, person_list_cor


def correlation_analysis(data):
    assmble = []
    assmble_cor = []
    random.shuffle(data)
    for i in range(len(data)):
        if i == len(data) - 1:
            person, person_cor = cal_person(data[i]['val'], data[0]['val'])
        else:
            person, person_cor = cal_person(data[i]['val'], data[i + 1]['val'])
        assmble.append(person)
        assmble_cor.append(person_cor)
    return assmble, assmble_cor


#
data_fccv = get_data(first_cluster_cell_voltage[:3])
# data_ectlc = get_data(each_cluster_total_loop_current)
# data_fcct = get_data(first_cluster_cell_temperature[:10])
fccv_res, fccv_res_cor = correlation_analysis(data_fccv)


# ectlc_res, ectlc_res_cor = correlation_analysis(data_ectlc)
# fcct_res, fcct_res_cor = correlation_analysis(data_fcct)


# %%
def plotys(data, names=None):
    fig = plt.figure()
    for i, r in enumerate(data):
        print(type(r))
        if names is None:
            label = 'data%d' % (i + 1)
        else:
            label = names[i]
        if type(r) == list:
            plt.plot(r, label=label)
        else:
            plt.plot(r['val'], label=label)
    plt.legend()
    plt.show()


# %%
plotys(fccv_res_cor)
plotys(data_fccv)
# %%
# plotnp(ectlc_res_cor)
# plotys(data_ectlc)
# %%
# plotnp(fcct_res_cor)
# plotys(data_fcct)
