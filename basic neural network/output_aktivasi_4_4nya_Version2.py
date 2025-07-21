import numpy as np

# Input dari -4 sampai 4
x = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def softplus(x):
    return np.log(1 + np.exp(x))

# Cetak output semua fungsi aktivasi
print("Input       :", x)
print("Sigmoid     :", np.round(sigmoid(x), 3))
print("Tanh        :", np.round(tanh(x), 3))
print("ReLU        :", relu(x))
print("Softplus    :", np.round(softplus(x), 3))