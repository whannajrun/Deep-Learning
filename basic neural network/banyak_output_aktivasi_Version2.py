import numpy as np

# Banyak input
x = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])

# Fungsi aktivasi
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

# Output fungsi aktivasi
print("Input:      ", x)
print("Linear:     ", x)
print("Sigmoid:    ", np.round(sigmoid(x), 3))
print("Tanh:       ", np.round(tanh(x), 3))
print("ReLU:       ", relu(x))