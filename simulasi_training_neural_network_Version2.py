import numpy as np
import matplotlib.pyplot as plt

# Data sederhana: y = 2x + 1 + noise
np.random.seed(0)
X = np.linspace(0, 1, 20)
y = 2 * X + 1 + 0.1 * np.random.randn(20)

# Inisialisasi bobot dan bias
w = np.random.randn()
b = np.random.randn()
lr = 0.1

# Simpan sejarah bobot dan loss
w_history, b_history, loss_history = [], [], []

# Fungsi prediksi dan loss
def predict(X, w, b):
    return w * X + b

def mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

# Training manual (gradient descent)
for epoch in range(40):
    y_pred = predict(X, w, b)
    loss = mse(y, y_pred)
    # Gradien
    dw = -2 * ((y - y_pred) * X).mean()
    db = -2 * (y - y_pred).mean()
    # Simpan sejarah
    w_history.append(w)
    b_history.append(b)
    loss_history.append(loss)
    # Update bobot
    w -= lr * dw
    b -= lr * db

    # Visualisasi progres setiap 10 epoch
    if epoch % 10 == 0 or epoch == 39:
        plt.figure(figsize=(5,3))
        plt.scatter(X, y, label='Data')
        plt.plot(X, y_pred, c='r', label=f'Prediksi epoch {epoch+1}')
        plt.title(f'Loss = {loss:.3f}')
        plt.legend()
        plt.show()

# Visualisasi loss dan bobot selama training
plt.figure(figsize=(10,3))
plt.subplot(1,3,1)
plt.plot(loss_history)
plt.title('Perubahan Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1,3,2)
plt.plot(w_history)
plt.title('Perubahan Bobot w')
plt.xlabel('Epoch')
plt.ylabel('w')

plt.subplot(1,3,3)
plt.plot(b_history)
plt.title('Perubahan Bias b')
plt.xlabel('Epoch')
plt.ylabel('b')
plt.tight_layout()
plt.show()