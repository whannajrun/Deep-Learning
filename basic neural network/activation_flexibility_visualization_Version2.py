import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# Buat data non-linear (lingkaran di dalam lingkaran)
X, y = make_circles(n_samples=300, factor=0.4, noise=0.15, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

# Model 1: Linear (tanpa fungsi aktivasi)
model_linear = MLPClassifier(hidden_layer_sizes=(10,), activation='identity', max_iter=2000, random_state=42)
model_linear.fit(X_scaled, y)

# Model 2: Non-linear (dengan ReLU)
model_relu = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', max_iter=2000, random_state=42)
model_relu.fit(X_scaled, y)

def plot_decision_boundary(clf, X, y, ax, title):
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300),
        np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 300)
    )
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolor='k')
    ax.set_title(title)
    ax.set_xlabel("Fitur 1")
    ax.set_ylabel("Fitur 2")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_decision_boundary(model_linear, X_scaled, y, axes[0], "Linear (Tanpa Fungsi Aktivasi)")
plot_decision_boundary(model_relu, X_scaled, y, axes[1], "Non-linear (Dengan ReLU)")
plt.tight_layout()
plt.show()