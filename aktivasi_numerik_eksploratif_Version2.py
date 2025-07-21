import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)

# Fungsi aktivasi
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

# Output fungsi aktivasi
y_sigmoid = sigmoid(x)
y_tanh    = tanh(x)
y_relu    = relu(x)
y_linear  = x  # sebagai pembanding

plt.figure(figsize=(8,6))
plt.plot(x, y_linear, '--', label='Linear (y=x)', color='gray')
plt.plot(x, y_sigmoid, label='Sigmoid')
plt.plot(x, y_tanh, label='Tanh')
plt.plot(x, y_relu, label='ReLU')
plt.title("Output berbagai fungsi aktivasi")
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()