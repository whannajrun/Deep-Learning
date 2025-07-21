import numpy as np
import matplotlib.pyplot as plt

# Data sederhana
X = np.linspace(0, 10, 100)
y = 2 * X + 1 + np.random.randn(100)  # garis miring, plus sedikit noise

# 1. Regresi Linear (statistik klasik)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X.reshape(-1, 1), y)
y_pred_lr = lr.predict(X.reshape(-1, 1))

# 2. Neural Network Sederhana (1 neuron, tanpa aktivasi)
from tensorflow import keras
model = keras.Sequential([
    keras.layers.Dense(1, input_shape=[1], activation=None)
])
model.compile(optimizer='sgd', loss='mse')
model.fit(X, y, epochs=100, verbose=0)
y_pred_nn = model.predict(X)

# Visualisasi
plt.scatter(X, y, label='Data')
plt.plot(X, y_pred_lr, label='Regresi Linear', color='red')
plt.plot(X, y_pred_nn, label='Neural Network', color='green', linestyle='dashed')
plt.legend()
plt.title('Regresi Linear vs Neural Network Sederhana')
plt.show()