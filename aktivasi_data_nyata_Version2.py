import numpy as np

# Contoh 'data nyata': bisa hasil dari layer sebelumnya, atau data input asli
# Di sini kita coba array acak dan sebagian angka yang merepresentasikan fitur nyata (misal, suhu, berat, skor)
data_riil = np.array([-2.5, -1.1, 0.0, 0.9, 2.3, 3.7, -3.2, 1.5, 0.2, -0.7])

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
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi)*(x + 0.044715*np.power(x,3))))

print("Data nyata      :", data_riil)
print("Sigmoid         :", np.round(sigmoid(data_riil), 4))
print("Tanh            :", np.round(tanh(data_riil), 4))
print("ReLU            :", np.round(relu(data_riil), 4))
print("Softplus        :", np.round(softplus(data_riil), 4))
print("LeakyReLU       :", np.round(leaky_relu(data_riil), 4))
print("GELU            :", np.round(gelu(data_riil), 4))