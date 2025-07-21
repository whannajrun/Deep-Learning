import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Buat data 2 kelas, bentuk "lingkaran di dalam lingkaran" (non-linear)
from sklearn.datasets import make_circles
X, y = make_circles(n_samples=400, factor=0.5, noise=0.05, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model 1: Neural Network TANPA fungsi aktivasi (hanya linear)
model_linear = MLPClassifier(
    hidden_layer_sizes=(8,), activation="identity", max_iter=1000, random_state=42
)
model_linear.fit(X_scaled, y)

# Model 2: Neural Network DENGAN fungsi aktivasi (ReLU)
model_nonlinear = MLPClassifier(
    hidden_layer_sizes=(8,), activation="relu", max_iter=1000, random_state=42
)
model_nonlinear.fit(X_scaled, y)

# Visualisasi decision boundary
def plot_decision_boundary(clf, X, y, ax, title):
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min()-0.5, X[:, 0].max()+0.5, 300),
        np.linspace(X[:, 1].min()-0.5, X[:, 1].max()+0.5, 300)
    )
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=18, edgecolors='k')
    ax.set_title(title)
    ax.set_xlabel("Fitur 1")
    ax.set_ylabel("Fitur 2")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_decision_boundary(model_linear, X_scaled, y, axes[0], "Tanpa Fungsi Aktivasi (Linear)")
plot_decision_boundary(model_nonlinear, X_scaled, y, axes[1], "Dengan Fungsi Aktivasi (ReLU)")
plt.tight_layout()
plt.show()