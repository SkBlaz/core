from neural import *
from clustering_reduction import *
from subspacereduction import *
try:
    import umap
except:
    print("UMAP unavailable.")
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import LocallyLinearEmbedding
import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger(__name__).setLevel(logging.INFO)


class CoRe:
    """
    The main CoRe class.
    """
    def __init__(self,
                 tau=2,
                 verbose=True,
                 embedding_algorithm="CoRe-small",
                 store_intermediary=False):
        self.verbose = verbose
        self.store_intermediary = store_intermediary
        self.intermediary_representations = []
        self.k = tau
        self.rep_scores = []
        self.embedding_algorithm = embedding_algorithm
        if "-direct" in self.embedding_algorithm:
            self.direct_projection = True
        else:
            self.direct_projection = False

    def dimension_series(self, max_y):
        dim_series = []
        cdim = max_y
        while cdim > self.k:
            temp_dim = int(cdim / self.k)
            if temp_dim > self.k:
                dim_series.append(temp_dim)
                cdim = temp_dim
            else:
                break
        dim_series.append(self.k)
        self.dim_series = dim_series
        logging.info("Initialized dimension series: {}".format(
            self.dim_series))

    def measure_complexity(self, matrix):
        norms = np.sqrt(np.einsum('ij,ij->i', matrix,
                                  matrix)) / matrix.shape[1]
        mnorm = (norms - np.mean(norms)) / (np.max(norms) - np.min(norms))
        mnorm = np.std(mnorm)
        return mnorm

    def fit(self, dataframe):
        dimension_y = min(dataframe.shape[1], dataframe.shape[0])
        self.dimension_series(dimension_y)
        encoders = []
        intermediary_representations = [dataframe]

        for dim in self.dim_series:
            if self.verbose:
                logging.info(f"Re-embedding into {dim} dimensions.")

            if "CoRe-large" in self.embedding_algorithm:
                encoder = GenericAutoencoder(n_components=dim,
                                             verbose=self.verbose,
                                             nn_type="large")

            if "CoRe-small" in self.embedding_algorithm:
                encoder = GenericAutoencoder(n_components=dim,
                                             verbose=self.verbose,
                                             nn_type="mini")

            elif "UMAP" in self.embedding_algorithm:
                encoder = umap.UMAP(n_components=dim)

            elif "RandomSubspace" in self.embedding_algorithm:
                encoder = ReductionSubspace(n_components=dim)

            elif "SparseRandom" in self.embedding_algorithm:
                encoder = SparseRandomProjection(n_components=dim)

            elif "NMF" in self.embedding_algorithm:
                encoder = NMF(n_components=dim)

            elif "Cluster-mean" in self.embedding_algorithm:
                encoder = ReductionCluster(n_dim=dim, aggregation="mean")

            elif "Cluster-median" in self.embedding_algorithm:
                encoder = ReductionCluster(n_dim=dim, aggregation="median")

            elif "Cluster-max" in self.embedding_algorithm:
                encoder = ReductionCluster(n_dim=dim, aggregation="max")

            elif "LLE" in self.embedding_algorithm:
                encoder = LocallyLinearEmbedding(n_components=dim)

            elif "PCA" in self.embedding_algorithm:
                encoder = TruncatedSVD(n_components=dim)

            ## encode the initial representation
            if self.direct_projection:
                encoded_representation = encoder.fit_transform(
                    intermediary_representations[0])

            ## encode current representation
            else:
                encoded_representation = encoder.fit_transform(
                    intermediary_representations[-1])

            self.rep_scores.append(
                self.measure_complexity(encoded_representation))

            encoders.append(encoder)
            intermediary_representations.append(encoded_representation)

        if self.store_intermediary:
            self.intermediary_representations = intermediary_representations
        self.encoder_space = encoders

    def transform(self, dataframe, keep_intermediary=True):

        current_df = dataframe
        if self.verbose:
            logging.info("Encoding new data.")

        if keep_intermediary:
            intermediary_representations = [dataframe]

        for encoder in self.encoder_space:
            tmp_df = encoder.transform(current_df)
            if keep_intermediary:
                intermediary_representations.append(tmp_df)
            if self.direct_projection:
                current_df = dataframe
            else:
                current_df = tmp_df

        if self.verbose:
            logging.info("Encoding obtained.")

        if keep_intermediary:
            return intermediary_representations
        else:
            return current_df

    def fit_transform(self, dataframe, keep_intermediary=False):

        self.fit(dataframe)
        return self.transform(dataframe, keep_intermediary)


if __name__ == "__main__":

    import numpy as np

    X = np.random.random((100, 100))
    ra_instance = CoRe(verbose=False,
                       embedding_algorithm="CoRe-small",
                       store_intermediary=False)
    ra_instance.fit(X)
    intermediary = ra_instance.transform(X, keep_intermediary=True)
    print(len(intermediary))
