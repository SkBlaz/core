# The clustering reduction class
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import normalize, StandardScaler


class ReductionSubspace:
    def __init__(self, n_components=2, normalization="l2"):
        self.n_dim = n_components
        self.normalize = normalization

    def fit(self, X):
        pass

    def transform(self, X):
        r_subset = np.random.choice(list(range(X.shape[1])), self.n_dim)
        fspace = X[:, r_subset]
        if self.normalize == "x":
            pass
        else:
            fspace = normalize(fspace, norm=self.normalize)
        return fspace

    def fit_transform(self, X):

        self.fit(X)
        return self.transform(X)


if __name__ == "__main__":

    clf = ReductionSubspace()
    X = np.random.random((10, 1000))
    transformed = clf.fit_transform(X)
    print(transformed.shape)
