import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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

# Rassal Orman modelini oluşturma ve eğitme
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Test verisi ile tahmin yapma
y_pred = clf.predict(X_test)

# Sonuçları değerlendirme
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Önemli özellikleri görselleştirme
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = iris.feature_names

plt.figure(figsize=(10, 6))
plt.title("Özelliklerin Önemi")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices])
plt.xlabel("Özellikler")
plt.ylabel("Önem Skoru")
plt.show()

# Sınıflandırma sonuçlarını görselleştirme (ilk iki özellik kullanılarak)
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', marker='o', edgecolor='k', s=100)
plt.title('Random Forest Sınıflandırma Sonuçları')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()
