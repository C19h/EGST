import numpy as np

from sklearn.preprocessing import StandardScaler

x = np.arange(10, dtype = np.float32)
x = x / 2 + 10
a = np.arange(10, dtype = np.float32) + 10
scaler = StandardScaler()

x_train = scaler.fit_transform(a[:, np.newaxis])

x_2 = scaler.transform(x[:, np.newaxis])
b = np.mean(np.abs((x - a)) / a)
print(b)
x_2 = x_2[:,0]
x_train = x_train[:,0]
c = np.mean(np.abs((x_2  - x_train)) / x_train)
print(c)