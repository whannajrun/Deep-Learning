import numpy as np
import matplotlib.pyplot as plt

# Buat input "gambar" 5x5
image = np.array([
    [1, 2, 0, 2, 1],
    [0, 1, 2, 1, 0],
    [1, 0, 1, 0, 2],
    [2, 1, 0, 2, 1],
    [0, 2, 1, 1, 0]
])

# Buat filter/kernerl 3x3
kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

# Fungsi konvolusi manual (tanpa padding, stride=1)
def conv2d_manual(img, kern):
    h_img, w_img = img.shape
    h_k, w_k = kern.shape
    h_out = h_img - h_k + 1
    w_out = w_img - w_k + 1
    output = np.zeros((h_out, w_out))
    # Sliding window
    for i in range(h_out):
        for j in range(w_out):
            region = img[i:i+h_k, j:j+w_k]
            output[i, j] = np.sum(region * kern)
    return output

# Proses konvolusi
feature_map = conv2d_manual(image, kernel)

# Visualisasi input, kernel, dan hasil
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Input (Gambar 5x5)")
plt.imshow(image, cmap='gray')
plt.colorbar()

plt.subplot(1,3,2)
plt.title("Kernel/Filter 3x3")
plt.imshow(kernel, cmap='gray')
plt.colorbar()

plt.subplot(1,3,3)
plt.title("Hasil Konvolusi (Feature Map)")
plt.imshow(feature_map, cmap='gray')
plt.colorbar()

plt.tight_layout()
plt.show()