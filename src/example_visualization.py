# Simple cluster visualization
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

from CoRe import CoRe

import matplotlib.pyplot as plt
import seaborn as sns

import umap
import numpy as np

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess

from class_clustering_reduction import ReductionCluster

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
    possible_embedding = "SVD"
    sns.set_style("white")
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    core_instance = CoRe(verbose=False,
                         embedding_algorithm=possible_embedding,
                         store_intermediary=True)
    core_instance.fit(frep)
    intermediary = core_instance.intermediary_representations
    
    for enx, transformed in enumerate(intermediary):
        ax = fig.add_subplot(1, len(intermediary), enx + 1)
        compressed_rep = umap.UMAP().fit_transform(transformed)

        sns.scatterplot(compressed_rep[:, 0],
                        compressed_rep[:, 1],
                        hue=labels,
                        palette="Accent",
                        s=12)

        plt.legend([], [], frameon=False)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.title(f"dim: {transformed.shape[1]}", fontsize=8)
    plt.tight_layout()
    plt.savefig("figures/grid.pdf", dpi=300)
