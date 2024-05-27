import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Adım 1: Veri kümesi oluşturun
# Rastgele bir veri kümesi oluşturmak için make_blobs kullanıyoruz
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Adım 2: K-means modelini oluşturun ve veriye fit edin
# KMeans modelini oluşturuyoruz ve belirli bir k (küme sayısı) belirliyoruz
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

# Adım 3: Sonuçları görselleştirin
# Veri noktalarını ve merkezlerini (centroid) çiziyoruz
plt.scatter(X[:, 0], X[:, 1], s=50, c=kmeans.labels_, cmap='viridis')

# Küme merkezlerini çiziyoruz
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title('K-means Kümeleme')
plt.xlabel('Özellik 1')
plt.ylabel('Özellik 2')
plt.show()
