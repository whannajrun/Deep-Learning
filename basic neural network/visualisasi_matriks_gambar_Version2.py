import numpy as np
import matplotlib.pyplot as plt

# Matriks gambar 4x4 (grayscale)
gambar = np.array([
    [0,   50, 200, 255],
    [10, 100, 180, 240],
    [30, 120, 160, 220],
    [80, 140, 150, 210]
])

plt.figure(figsize=(3,3))
plt.imshow(gambar, cmap='gray', vmin=0, vmax=255)
plt.title("Visualisasi Gambar 4x4 (Grayscale)")
plt.colorbar(label="Intensitas (0=hitam, 255=putih)")
plt.show()

# Tampilkan juga matrix-nya
print("Matriks gambar (angka intensitas pixel):\n", gambar)