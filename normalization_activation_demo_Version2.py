import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

# Contoh data: dua fitur, dua kelas (klasifikasi sederhana)
np.random.seed(42)
X = np.random.rand(100, 2) * 20 + 40   # nilai antara 40 dan 60
y = (X[:, 0] + X[:, 1] > 100).astype(int)  # label 0 atau 1

# --- Normalisasi (Standardization) ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Neural network sederhana ---
model = keras.Sequential([
    keras.layers.Input(shape=(2,)),
    keras.layers.Dense(4, activation='relu'),  # Fungsi aktivasi di sini!
    keras.layers.Dense(1, activation='sigmoid') # Output: probabilitas (fungsi aktivasi sigmoid)
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_scaled, y, epochs=50, verbose=0)

# --- Visualisasi output sebelum dan sesudah normalisasi ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title("Data sebelum dinormalisasi")
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel("Fitur 1")
plt.ylabel("Fitur 2")

plt.subplot(1, 2, 2)
plt.title("Data setelah dinormalisasi")
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y)
plt.xlabel("Fitur 1 (scaled)")
plt.ylabel("Fitur 2 (scaled)")
plt.tight_layout()
plt.show()