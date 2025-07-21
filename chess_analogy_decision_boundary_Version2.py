import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# Buat data 'moons' (non-linear)
X, y = make_moons(n_samples=300, noise=0.23, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

# Model pion (tanpa fungsi aktivasi / linear)
model_pion = MLPClassifier(hidden_layer_sizes=(8,), activation='identity', max_iter=2000, random_state=42)
model_pion.fit(X_scaled, y)

# Model ratu (dengan fungsi aktivasi ReLU / non-linear)
model_ratu = MLPClassifier(hidden_layer_sizes=(8,), activation='relu', max_iter=2000, random_state=42)
model_ratu.fit(X_scaled, y)

# Visualisasi decision boundary
def plot_decision_boundary(clf, X, y, ax, title, color):
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min()-0.5, X[:, 0].max()+0.5, 300),
        np.linspace(X[:, 1].min()-0.5, X[:, 1].max()+0.5, 300)
    )
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolor='k', s=18)
    ax.set_title(title)
    ax.set_xlabel("Fitur 1")
    ax.set_ylabel("Fitur 2")
    # Tambahan gambar langkah pion/ratu (ilustrasi)
    if color == "pion":
        ax.plot([0, 0], [-2, 2], "k--", linewidth=3, label="Langkah pion (lurus)")
    if color == "ratu":
        circle = plt.Circle((0, 0), 1, color="g", fill=False, linewidth=2, label="Langkah ratu (fleksibel)")
        ax.add_artist(circle)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_decision_boundary(model_pion, X_scaled, y, axes[0], "Tanpa Fungsi Aktivasi (Pion: Lurus)", "pion")
plot_decision_boundary(model_ratu, X_scaled, y, axes[1], "Dengan Fungsi Aktivasi (Ratu: Fleksibel)", "ratu")
axes[0].legend()
axes[1].legend()
plt.tight_layout()
plt.show()