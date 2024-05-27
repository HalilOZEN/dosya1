import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Iris veri setini yükleme
iris = load_iris()
X = iris.data

# Veriyi standardize etme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means modelini oluşturma ve eğitme
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Küme etiketlerini tahmin etme
y_kmeans = kmeans.predict(X_scaled)

# PCA ile veriyi 2 boyuta indirgeme (görselleştirme için)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Kümeleme sonuçlarını görselleştirme
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis', marker='o', edgecolor='k', s=100)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Cluster Centers')
plt.title('K-Means Kümeleme Sonuçları')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.show()
