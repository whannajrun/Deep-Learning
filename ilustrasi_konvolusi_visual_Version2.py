import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Gambar 5x5
image = np.array([
    [1, 2, 0, 2, 1],
    [0, 1, 2, 1, 0],
    [1, 0, 1, 0, 2],
    [2, 1, 0, 2, 1],
    [0, 2, 1, 1, 0]
])

# Filter 3x3
kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

def visualize_convolution(img, kern, step=0):
    fig, ax = plt.subplots(1)
    ax.imshow(img, cmap='gray', vmin=0, vmax=2)
    h_k, w_k = kern.shape
    # Highlight area yang diproses
    rect = patches.Rectangle((step[1]-0.5, step[0]-0.5), w_k, h_k, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    # Tampilkan nilai di tiap sel
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ax.text(j, i, str(img[i, j]), va='center', ha='center', color='blue', fontsize=14)
    plt.title(f'Window di posisi {step}')
    plt.axis('off')
    plt.show()

def konvolusi_dengan_visual(img, kern):
    h_img, w_img = img.shape
    h_k, w_k = kern.shape
    h_out = h_img - h_k + 1
    w_out = w_img - w_k + 1
    hasil = np.zeros((h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            visualize_convolution(img, kern, (i, j))
            region = img[i:i+h_k, j:j+w_k]
            print(f"Region yang diambil:\n{region}")
            print(f"Kernel:\n{kern}")
            print(f"Region * Kernel:\n{region * kern}")
            hasil[i, j] = np.sum(region * kern)
            print(f"Hasil penjumlahan: {hasil[i, j]}\n{'-'*40}")
    return hasil

# Jalankan konvolusi dengan visualisasi
feature_map = konvolusi_dengan_visual(image, kernel)

# Visualisasi hasil akhir
plt.imshow(feature_map, cmap='gray')
plt.title('Feature Map hasil Konvolusi')
plt.colorbar()
plt.show()