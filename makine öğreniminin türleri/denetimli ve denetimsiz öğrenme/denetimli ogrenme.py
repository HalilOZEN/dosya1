import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Iris veri setini yükleme
iris = load_iris()
X, y = iris.data, iris.target

# Veri setini eğitim ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi standardize etme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Lojistik regresyon modelini oluşturma ve eğitme
clf = LogisticRegression(random_state=42, max_iter=200)
clf.fit(X_train, y_train)

# Test verisi ile tahmin yapma
y_pred = clf.predict(X_test)

# Sonuçları değerlendirme
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Sınıflandırma sonuçlarını görselleştirme (ilk iki özellik kullanılarak)
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', marker='o', edgecolor='k', s=100)
plt.title('Lojistik Regresyon Sınıflandırma Sonuçları')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()
