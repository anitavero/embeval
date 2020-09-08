import os
import argh
from argh import arg
import re
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_distances
from nltk.corpus import wordnet as wn
from itertools import chain
from tqdm import tqdm
import json
from source.utils import suffixate

from source.process_embeddings import Embeddings


################# Nearest Neigbor metrics #################


def get_n_nearest_neighbors(words: np.ndarray, E: np.ndarray, vocab: np.ndarray, n: int = 10):
    """n nearest neighbors for words based on cosine distance in Embedding E."""
    w_idx = np.where(np.in1d(vocab, np.array(words)))[0]   # Words indices in Vocab and Embedding E
    C = cosine_distances(E)
    np.fill_diagonal(C, np.inf)
    w_C = C[:, w_idx]                                      # Filter columns for words
    nNN = np.argpartition(w_C, range(n), axis=0)[:n]       # Every column j contains the indices of NNs of word_j
    return np.vstack([words, vocab[nNN]])                  # 1st row: words, rows 1...n: nearest neighbors


@arg('-w', '--words', nargs='+', type=str)
def n_nearest_neighbors(data_dir, model_name, words=[], n: int = 10):
    """n nearest neighbors for words based on model <vecs_names>."""
    embs = Embeddings(data_dir, [model_name])
    E, vocab = embs.embeddings[0], embs.vocabs[0]
    return get_n_nearest_neighbors(np.array(words), E, vocab, n).transpose()


################# Clusterization metrics #################


def dbscan_clustering(model, eps=0.5, min_samples=90, n_jobs=4):
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine', n_jobs=n_jobs).fit(model)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    return labels


def kmeans(model, n_clusters=3, random_state=1, n_jobs=4):
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=random_state, verbose=True,
                          n_jobs=n_jobs).fit(model)
    labels = kmeans_model.labels_
    return labels


def cluster_eval(vectors, labels):
    """Unsupervised clustering metrics."""
    results = {}
    def safe_metric(metric):
        name = re.sub('_', ' ', metric.__name__).title()
        try:
            if metric == metrics.silhouette_score:
                results[name] = round(metric(vectors, labels, metric='cosine'), 4)
            else:
                results[name] = round(metric(vectors, labels), 4)
        except ValueError as e:
            print("[{0}] {1}".format(name, e))

    safe_metric(metrics.silhouette_score)
    safe_metric(metrics.calinski_harabaz_score)
    safe_metric(metrics.davies_bouldin_score)

    return results


def run_clustering(model_file, cluster_method, n_clusters=3, random_state=1, eps=0.5, min_samples=90,
                   workers=4):
    if model_file == 'random':
        model = np.random.random(size=(70000, 300))
    else:
        model = np.load(model_file)

    if cluster_method == 'dbscan':
        labels = dbscan_clustering(model, eps=eps, min_samples=min_samples, n_jobs=workers)
    elif cluster_method == 'kmeans':
        labels = kmeans(model, n_clusters=n_clusters, random_state=random_state, n_jobs=workers)

    return cluster_eval(model, labels)


@arg('-mns', '--model_names', nargs='+', type=str, required=True)
def run_clustering_experiments(datadir='/anfs/bigdisc/alv34/wikidump/extracted/models/',
                               savedir='/anfs/bigdisc/alv34/wikidump/extracted/models/results/',
                               model_names=[], cluster_method='dbscan', n_clusters=3, random_state=1,
                               eps=0.5, min_samples=90, workers=4, suffix=''):
    for m in tqdm(model_names):
        mname = m.split('.')[0]
        print(mname)
        model_metrics = run_clustering(os.path.join(datadir, m), cluster_method, n_clusters, random_state,
                                          eps, min_samples, workers)
        with open(os.path.join(savedir, f'cluster_metrics_{cluster_method}_{mname}{suffixate(suffix)}.json'), 'w') as f:
            json.dump(model_metrics, f)


def wn_category(word):
    """Map a word to categories based on WordNet closures."""
    cats = ['transport', 'food', 'building', 'animal', 'appliance', 'action', 'clothes', 'utensil', 'body', 'color',
            'electronics', 'number', 'human']
    cat_synsets = dict(zip(cats, map(wn.synsets, cats)))
    hyper = lambda s: s.hypernyms()
    synsets = wn.synsets(word)
    closures = list(chain.from_iterable([list(sns.closure(hyper, depth=3)) for sns in synsets])) + synsets
    max_overlap = 0
    category = None
    for cat, csns in cat_synsets.items():
        if len(set(closures).intersection(set(csns))) > max_overlap:
            category = cat
    return category


if __name__ == '__main__':
    argh.dispatch_commands([run_clustering, run_clustering_experiments, n_nearest_neighbors])
    # vocab = np.array(['a', 'b', 'c', 'd', 'e'])
    # words = np.array(['a', 'c', 'e'])
    # E = np.array([[1, 0],
    #               [0, 1],
    #               [1, 1],
    #               [1, 0.5],
    #               [0.5, 1]])
    #
    #
    # NN = get_n_nearest_neighbors(words, E, vocab, n=1)
    # assert (NN[0, :] == words).all()
    # assert (NN[1:, 0] == np.array(['d'])).all()
    # assert (NN[1:, 1] == np.array(['d'])).all()