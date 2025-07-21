import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Data dummy dari file sebelumnya
X = np.array([
    [0.2, 0.1, 0.5, 0.7],
    [0.8, 0.9, 0.1, 0.4],
    [0.3, 0.2, 0.6, 0.8],
    [0.7, 0.8, 0.2, 0.3],
    [0.1, 0.2, 0.5, 0.6],
    [0.9, 0.8, 0.2, 0.1],
])
y = np.array([0, 1, 0, 1, 0, 2])

# Split data train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Model neural network sederhana
clf = MLPClassifier(hidden_layer_sizes=(8,), activation='relu', solver='adam', max_iter=500, random_state=42)
clf.fit(X_train, y_train)

# Evaluasi
y_pred = clf.predict(X_test)
print("Klasifikasi Test:")
print(classification_report(y_test, y_pred, target_names=['T-shirt', 'Celana', 'Sepatu']))