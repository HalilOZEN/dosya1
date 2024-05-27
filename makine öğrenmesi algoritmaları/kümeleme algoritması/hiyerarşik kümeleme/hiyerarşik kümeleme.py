import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.datasets import make_blobs

# Örnek veri seti oluşturma
X, _ = make_blobs(n_samples=50, centers=3, random_state=42)

# Hiyerarşik kümeleme (linkage)
Z = linkage(X, method='ward')

# Dendrogram çizimi
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title('Hiyerarşik Kümeleme Dendrogramı')
plt.xlabel('Örnek İndeksi')
plt.ylabel('Mesafe')
plt.show()

# Kümeleri belirli bir mesafe eşiğine göre ayırma
max_d = 7.0  # Mesafe eşiği
clusters = fcluster(Z, max_d, criterion='distance')

# Küme sonuçlarını görselleştirme
plt.figure(figsize=(10, 7))
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='prism')
plt.title('Hiyerarşik Kümeleme Sonucu')
plt.xlabel('Özellik 1')
plt.ylabel('Özellik 2')
plt.show()
