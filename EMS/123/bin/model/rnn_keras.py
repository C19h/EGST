import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def mape(Y_true, Y_pred):
    return keras.metrics.mean_absolute_percentage_error(Y_true[:, -1], Y_pred[:, -1])

class Model():
    def __init__(self):
        pass

    def build(self, timestep = 7, loss = 'mse', lr = 0.0001, decay = 0.001):
        inputs = keras.layers.Input(shape=[None, 2])
        rnnhiden = keras.layers.LSTM(100, activation='relu', return_sequences=True, dropout=0.1)(inputs)
        rnnhiden = keras.layers.LSTM(100, activation='relu', return_sequences=True, dropout=0.1)(rnnhiden)
        output = keras.layers.TimeDistributed(keras.layers.Dense(1, activation='relu'))(rnnhiden)
        model = keras.Model(inputs=inputs, outputs=output)
        model.summary()
        model.compile(loss="mse", optimizer=keras.optimizers.Nadam(learning_rate=lr, decay=decay),
            metrics=[mape])
        self.model = model
        self.timestep = timestep
        return model

    def train(self, datax, datay, epoches = 400, trainrate = 0.8):
        if isinstance(datax, list):
            X = np.concatenate([datax[0].reshape(-1, 1), datax[1].reshape(-1, 1)], axis=1)
            for i in range(2, len(datax)):
                X = np.concatenate([X, datax[1].reshape(-1, 1)], axis=1)

        self.scaler = StandardScaler()
        SPLITNUM = int(X.shape[0]*trainrate) - self.timestep
        self.scaler.fit(X[:SPLITNUM, :])
        X = self.scaler.transform(X)
        x_train_all = np.zeros((X.shape[0] - self.timestep, self.timestep, X.shape[1]))
        y_train_all = np.zeros((X.shape[0] - self.timestep, self.timestep))
        for i in range(0, X.shape[0] - self.timestep):
            x_train_all[i, :, :] = X[i:i + self.timestep, :]
            y_train_all[i, ...] = datay[i + 1:i + self.timestep + 1]
        y_train_all = y_train_all.reshape(-1, self.timestep, 1)
        x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=1 - trainrate, shuffle=False)
        
        checkpoint_cb = keras.callbacks.ModelCheckpoint("./my_zm_model5.h5", save_best_only=True)
        self.model.fit(x_train, y_train, epochs=epoches,
            validation_data=(x_test, y_test), batch_size=50,
            callbacks=[checkpoint_cb, keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True)]
            )

        res = self.model.predict(x_test)
        self._plot([res[:,-1,0], y_test[:,-1,0]])
        print(res)

if __name__ == "__main__":
    import sys, os, math
    sys.path.append('./bin/model')
    from zm_jlzx_hour import ZM_Jlzx_Hour
    zj = ZM_Jlzx_Hour()
    zj.load()
    wdata = zj.get_pddata_workday()
    x0_1, x0_2, y = zj.prepare_hour_workday(wdata)
    model = Model()
    model.build()
    model.train([x0_1,x0_2], y, 400)