import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# Buat data non-linear (moons)
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

# Model 1: Neural Network tanpa fungsi aktivasi (linear, activation="identity")
model_linear = MLPClassifier(hidden_layer_sizes=(8,), activation="identity", max_iter=2000, random_state=1)
model_linear.fit(X_scaled, y)

# Model 2: Neural Network dengan fungsi aktivasi (ReLU)
model_relu = MLPClassifier(hidden_layer_sizes=(8,), activation="relu", max_iter=2000, random_state=1)
model_relu.fit(X_scaled, y)

# Fungsi visualisasi decision boundary
def plot_decision_boundary(model, X, y, ax, title):
    xx, yy = np.meshgrid(
        np.linspace(X[:,0].min() - 0.5, X[:,0].max() + 0.5, 300),
        np.linspace(X[:,1].min() - 0.5, X[:,1].max() + 0.5, 300)
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    ax.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.coolwarm, edgecolor='k')
    ax.set_title(title)
    ax.set_xlabel("Fitur 1")
    ax.set_ylabel("Fitur 2")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_decision_boundary(model_linear, X_scaled, y, axes[0], "Tanpa Fungsi Aktivasi (Linear)")
plot_decision_boundary(model_relu, X_scaled, y, axes[1], "Dengan Fungsi Aktivasi (ReLU)")
plt.tight_layout()
plt.show()