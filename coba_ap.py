print(__doc__)

from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AdapAffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn import datasets


# #############################################################################
# Generate sample data
"""
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=1000, centers=centers, cluster_std=0.5,
                            random_state=0)
"""

data_wine = datasets.load_wine()
#digits = datasets.load_digits()
iris = datasets.load_iris()
"""
X = digits.data
labels_true = digits.target

X = iris.data
labels_true = iris.target

"""
X = data_wine.data
labels_true = data_wine.target

#print(X)
# #############################################################################
# Compute Affinity Propagation
#af = AffinityPropagation(preference=-50).fit(X)
#af = AffinityPropagation().fit(X)
af = AdapAffinityPropagation().fit(X)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)

print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
#score between 0.0 and 1.0. 1.0 stands for perfectly homogeneous labeling

print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
#score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling

print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
#score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling

print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
#Similarity score between -1.0 and 1.0. Random labelings have an ARI close to 0.0. 1.0 stands for perfect match.

print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
#The AMI returns a value of 1 when the two partitions are identical(ie perfectly matched).

print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels, metric='sqeuclidean'))
#The best value is 1 and the worst value is -1.

#print("exemplars : ", cluster_centers_indices)
print("iteration : ", af.n_iter_)
# #############################################################################

# Plot result

import matplotlib.pyplot as plt
from itertools import cycle

plt.close('all')
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = X[cluster_centers_indices[k]]
    plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    for x in X[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

