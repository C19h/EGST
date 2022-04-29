# author:c19h
# datetime:2022/4/19 10:27
import matplotlib.pyplot as plt
import numpy as np


def plotys(data, names=None):
    fig = plt.figure()
    for i in range(len(data)):
        if names is None:
            label = 'data%d' % (i + 1)
        else:
            label = names[i]
        plt.plot(data.iloc[i], label=label)
    plt.legend()
    plt.show()
