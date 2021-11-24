## The clustering reduction class
from sklearn.cluster import KMeans
import numpy as np


class ReductionCluster:
    def __init__(self, n_dim=2, aggregation="max"):
        self.n_dim = n_dim
        self.aggregation = aggregation

    def fit(self, X):
        X = X.T
        self.clf = KMeans(n_clusters=self.n_dim).fit(X)
        self.labels = self.clf.labels_

    def transform(self, X):
        unique_labels = np.unique(self.labels)
        rspace = np.zeros((X.shape[0], len(unique_labels)))
        for enx, j in enumerate(unique_labels):
            subspace = X[:, np.where(self.labels == j)[0]]
            if self.aggregation == "max":
                rspace[:, enx] = np.max(subspace, axis=1).reshape(-1)
            elif self.aggregation == "mean":
                rspace[:, enx] = np.mean(subspace, axis=1).reshape(-1)
            elif self.aggregation == "median":
                rspace[:, enx] = np.mean(subspace, axis=1).reshape(-1)
        return rspace

    def fit_transform(self, X):

        self.fit(X)
        return self.transform(X)
