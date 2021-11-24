## Simple cluster visualization
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from CoRe import *
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import umap
import numpy as np
from gensim.utils import simple_preprocess
from sklearn.cluster import KMeans
from clustering_reduction import ReductionCluster

if __name__ == "__main__":

    train_df = "data/CNN-news/train.tsv"
    train_df = pd.read_csv(train_df, sep="\t")
    texts_train = train_df.text_a
    labels = train_df.label.values.tolist()
    total_sequences_training = texts_train.values.tolist()
    documents = [
        TaggedDocument(simple_preprocess(doc), [i])
        for i, doc in enumerate(total_sequences_training)
    ]
    model = Doc2Vec(documents,
                    vector_size=768,
                    window=5,
                    min_count=2,
                    epochs=32,
                    num_cpu=8)
    vecs_train = []

    for doc in total_sequences_training:
        vector = model.infer_vector(simple_preprocess(doc))
        vecs_train.append(vector)

    print("Reduction")
    frep = np.array(vecs_train)
    fig = plt.figure(figsize=(6, 2))
    sns.set(font_scale=0.4)
    possible_embedding = "PCA"
    sns.set_style("white")
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for enx, ndim in enumerate([2, 4, 16, 64, 128]):
        ax = fig.add_subplot(1, 5, enx + 1)
        clf = TruncatedSVD(n_components=ndim)
        transformed = clf.fit_transform(frep)
        print(transformed.shape)
        compressed_rep = umap.UMAP().fit_transform(transformed)

        sns.scatterplot(compressed_rep[:, 0],
                        compressed_rep[:, 1],
                        hue=labels,
                        palette="Accent",
                        s=12)

        plt.legend([], [], frameon=False)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.title(f"dim: {ndim}", fontsize=15)
    plt.tight_layout()
    plt.savefig("figures/grid.pdf", dpi=300)
