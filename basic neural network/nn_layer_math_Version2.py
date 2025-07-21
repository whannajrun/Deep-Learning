import numpy as np

# Input
x = np.array([[1], [2]])

# Bobot dan bias
W = np.array([[0.5, -1], [1, 2]])
b = np.array([[0], [1]])

# Hitung z = W.x + b
z = W @ x + b
print("z (sebelum aktivasi):", z.ravel())

# Fungsi aktivasi ReLU
a = np.maximum(0, z)
print("a (setelah aktivasi):", a.ravel())