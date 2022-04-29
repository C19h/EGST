from configmanager import ConfigManager
import matplotlib.pyplot as plt

def plotnp(npdata, names = None):
    plot = ConfigManager.Instance.GetConfigDefault('main', 'plot', '1')
    if plot != '1':
        pass
    else:
        for i, r in enumerate(npdata):
            if names is None:
                label = 'data%d' % (i+1)
            else:
                label = names[i]
            plt.plot(r, label=label)
        plt.legend()
        plt.show()