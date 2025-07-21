import numpy as np

# Input sederhana
x = np.array([-3, -1, 0, 1, 3])

# Fungsi aktivasi
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

# Hitung output
print("Input:", x)
print("Linear (y=x):      ", x)
print("Sigmoid:           ", np.round(sigmoid(x), 3))
print("Tanh:              ", np.round(tanh(x), 3))
print("ReLU:              ", relu(x))