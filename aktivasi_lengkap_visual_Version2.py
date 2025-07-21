import numpy as np
import matplotlib.pyplot as plt

# Input banyak nilai
x = np.linspace(-5, 5, 100)

# Definisi fungsi aktivasi
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def softplus(x):
    return np.log(1 + np.exp(x))

def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

def gelu(x):
    # Approximation of GELU
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi)*(x + 0.044715*np.power(x,3))))

# Hitung output untuk semua fungsi
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)
y_softplus = softplus(x)
y_leakyrelu = leaky_relu(x)
y_gelu = gelu(x)

# Visualisasi grafik
plt.figure(figsize=(10,7))
plt.plot(x, y_sigmoid, label="Sigmoid")
plt.plot(x, y_tanh, label="Tanh")
plt.plot(x, y_relu, label="ReLU")
plt.plot(x, y_softplus, label="Softplus")
plt.plot(x, y_leakyrelu, label="LeakyReLU")
plt.plot(x, y_gelu, label="GELU")
plt.title("Perbandingan Berbagai Fungsi Aktivasi")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.legend()
plt.show()

# Output numerik untuk input -4 sampai 4
x_num = np.arange(-4, 5)
print("Input      :", x_num)
print("Sigmoid    :", np.round(sigmoid(x_num), 3))
print("Tanh       :", np.round(tanh(x_num), 3))
print("ReLU       :", relu(x_num))
print("Softplus   :", np.round(softplus(x_num), 3))
print("LeakyReLU  :", np.round(leaky_relu(x_num), 3))
print("GELU       :", np.round(gelu(x_num), 3))

# Penjelasan pola output (ringkasan, bisa kamu baca di bawah ini)
penjelasan = """
Penjelasan pola output fungsi aktivasi:
- Sigmoid: Output antara 0 dan 1, perubahan S-curve. Sering untuk output probabilitas.
- Tanh: Output antara -1 sampai 1, S-curve dua arah. Centered di nol.
- ReLU: Output nol untuk input negatif, sama dengan input untuk positif. Sering dipakai di hidden layer.
- Softplus: Mirip ReLU tapi halus (tidak ada belokan).
- LeakyReLU: Seperti ReLU, tapi input negatif tidak langsung nol, melainkan dikali konstanta kecil (agar gradien tetap mengalir).
- GELU: Mirip sigmoid/tanh tapi lebih smooth, populer di model-model terbaru (misal: BERT).
"""
print(penjelasan)