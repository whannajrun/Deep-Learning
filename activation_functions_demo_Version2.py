import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 200)

# Fungsi aktivasi yang umum
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

plt.figure(figsize=(10, 6))
plt.plot(x, relu(x), label="ReLU (max(0, x))")
plt.plot(x, sigmoid(x), label="Sigmoid")
plt.plot(x, tanh(x), label="Tanh")
plt.title("Contoh Fungsi Aktivasi")
plt.xlabel("Input (x)")
plt.ylabel("Output f(x)")
plt.legend()
plt.grid(True)
plt.show()