import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


pts50 = 50 + np.random.randn(2, 50)
pts10 = 10 + np.random.randn(2, 50)

pts = np.concatenate([pts10, pts50], axis=1)

clustering = DBSCAN(eps=3, min_samples=2).fit(pts.T)
print(clustering)
print(clustering.labels_)


plt.scatter(pts[0],pts[1])
plt.show()

