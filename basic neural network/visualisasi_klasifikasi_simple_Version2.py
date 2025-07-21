import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ConfusionMatrixDisplay

# Data dummy: 3 kelas (0, 1, 2), 2 fitur (misal hasil ekstraksi gambar)
X = np.array([
    [0.2, 0.8],
    [0.1, 0.7],
    [0.25, 0.85],
    [0.9, 0.3],
    [0.8, 0.2],
    [0.85, 0.15],
    [0.4, 0.4],
    [0.45, 0.35],
    [0.5, 0.5]
])
y = np.array([0,0,0,1,1,1,2,2,2])

# Visualisasi data (sebelum training)
plt.figure(figsize=(6,6))
for label, marker, color in zip([0,1,2], ['o','s','^'], ['b','g','r']):
    plt.scatter(X[y==label,0], X[y==label,1], marker=marker, color=color, label=f'Kelas {label}')
plt.xlabel('Fitur 1')
plt.ylabel('Fitur 2')
plt.title('Visualisasi Data Dummy')
plt.legend()
plt.show()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Model neural network sederhana
clf = MLPClassifier(hidden_layer_sizes=(5,), activation='relu', random_state=42, max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Visualisasi hasil prediksi pada data test
plt.figure(figsize=(6,6))
for label, marker, color in zip([0,1,2], ['o','s','^'], ['b','g','r']):
    plt.scatter(X_test[y_pred==label,0], X_test[y_pred==label,1],
                marker=marker, color=color, label=f'Prediksi kelas {label}', edgecolor='k', s=100)
plt.xlabel('Fitur 1')
plt.ylabel('Fitur 2')
plt.title('Prediksi Kelas Data Test')
plt.legend()
plt.show()

# Visualisasi confusion matrix
ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, display_labels=['Kelas 0','Kelas 1','Kelas 2'])
plt.title('Confusion Matrix')
plt.show()