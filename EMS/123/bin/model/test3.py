import sys, os, math
#import keras
import numpy as np
sys.path.append('./bin/model')
from zm_jlzx_day import ZM_Jlzx_Day
from sklearn.model_selection import train_test_split
zj = ZM_Jlzx_Day()
zj.load()
wdata = zj.get_pddata_workday()
x0_1, x0_2, y = zj.prepare_day_workday(wdata)

yy = y[1:]
p =y[:-1]

q = np.average(np.abs(p -yy) /yy)
print(q)
zj._plot([yy, p])

# 模型1
x0_1 = x0_1.reshape(-1, 1)
x0_2 = x0_2.reshape(-1, 1)
X = np.concatenate([x0_1, x0_2], axis=1)
# %%
time_step = 7
x_train_all = np.zeros((len(X) - time_step, time_step, 2))
y_train_all = np.zeros((len(X) - time_step, time_step))
for i in range(0, len(X) - time_step):
    x_train_all[i, :, :] = X[i:i + time_step, :]
    y_train_all[i, ...] = y[i:i + time_step]
y_train_all = y_train_all.reshape(-1, time_step, 1)
x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, shuffle=False)


# %%
def last_time_step_mape(Y_true, Y_pred):
    return keras.metrics.mean_absolute_percentage_error(Y_true[:, -1], Y_pred[:, -1])


def mape(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true)) / np.mean(np.abs(y_true)) * 100


def get_model(n_neurons=50, learning_rate=0.0001):
    inputs = keras.layers.Input(shape=[None, 2])
    rnnhiden = keras.layers.LSTM(100, activation='relu', return_sequences=True, dropout=0.1)(inputs)
    rnnhiden = keras.layers.LSTM(100, activation='relu', return_sequences=True, dropout=0.1)(rnnhiden)
    output = keras.layers.TimeDistributed(keras.layers.Dense(1, activation='relu'))(rnnhiden)
    model = keras.Model(inputs=inputs, outputs=output)
    model.summary()
    model.compile(loss="mse", optimizer=keras.optimizers.Nadam(learning_rate=learning_rate, decay=0.001),
                    metrics=[last_time_step_mape])
    return model


# %%

rnn = get_model()
checkpoint_cb = keras.callbacks.ModelCheckpoint("./my_zm_model5.h5", save_best_only=True)
rnn.fit(x_train, y_train, epochs=400,
        validation_data=(x_test, y_test), batch_size=50,
        callbacks=[checkpoint_cb, keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True)])