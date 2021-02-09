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
from glob import glob
from tabulate import tabulate
import matplotlib
matplotlib.rcParams["savefig.dpi"] = 300
import matplotlib.pyplot as plt
matplotlib.style.use('fivethirtyeight')
from itertools import groupby
from collections import defaultdict

from source.utils import suffixate, tuple_list
from source.process_embeddings import Embeddings, mid_fusion, filter_by_vocab


FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figs')


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


def run_clustering(model, cluster_method, n_clusters=3, random_state=1, eps=0.5, min_samples=90,
                   workers=4):
    if type(model) == str:
        model = np.load(model)

    if cluster_method == 'dbscan':
        labels = dbscan_clustering(model, eps=eps, min_samples=min_samples, n_jobs=workers)
    elif cluster_method == 'kmeans':
        labels = kmeans(model, n_clusters=n_clusters, random_state=random_state, n_jobs=workers)

    return cluster_eval(model, labels), labels


@arg('-vns', '--vecs_names', nargs='+', type=str, required=True)
def get_clustering_labels_metrics(vecs_names=[], datadir='/anfs/bigdisc/alv34/wikidump/extracted/models/',
                                  savedir='/anfs/bigdisc/alv34/wikidump/extracted/models/results/',
                                  cluster_method='kmeans', n_clusters=3, random_state=1, eps=0.5, min_samples=90,
                                  workers=4, suffix=''):
    embs = Embeddings(datadir, vecs_names)
    for e, v, l in list(zip(embs.embeddings, embs.vocabs, embs.vecs_names)):
        print(l)
        model_metrics, cl_labels = run_clustering(e, cluster_method, n_clusters, random_state, eps, min_samples, workers)
        with open(os.path.join(savedir, f'cluster_metrics_labelled_{cluster_method}_{l}_nc{n_clusters}{suffixate(suffix)}.json'),
                  'w') as f:
            json.dump(model_metrics, f)

        with open(os.path.join(savedir, f'cluster_labels_{cluster_method}_{l}_nc{n_clusters}{suffixate(suffix)}.json'),
                  'w') as f:
            label_dict = {w: str(l) for l, w in zip(cl_labels, v)}
            json.dump(label_dict, f)


@arg('-mmembs', '--mm_embs_of', type=tuple_list)
@arg('-vns', '--vecs_names', nargs='+', type=str, required=True)
def run_clustering_experiments(datadir='/anfs/bigdisc/alv34/wikidump/extracted/models/',
                               savedir='/anfs/bigdisc/alv34/wikidump/extracted/models/results/',
                               vecs_names=[], mm_embs_of=[], cluster_method='dbscan', n_clusters=-1, random_state=1,
                               eps=0.5, min_samples=90, workers=4, suffix=''):
    # TODO: Test
    embs = Embeddings(datadir, vecs_names)
    emb_tuples = [tuple(embs.embeddings[vecs_names.index(l)] for l in t) for t in mm_embs_of]
    vocab_tuples = [tuple(embs.vocabs[vecs_names.index(l)] for l in t) for t in mm_embs_of]
    mm_labels = [tuple(l for l in t) for t in mm_embs_of]
    mm_embeddings, mm_vocabs, mm_labels = mid_fusion(emb_tuples, vocab_tuples, mm_labels, padding=False)

    models = []
    labels = []
    # Filter all with intersection fo vocabs
    isec_vocab = set.intersection(*map(set, mm_vocabs))
    with open(os.path.join(savedir, 'common_subset_vocab_VG_GoogleResnet_Wiki2020.json'), 'w') as f:
        json.dump(list(isec_vocab), f)
    print('#Common subset vocab:', len(isec_vocab))
    for e, v, l in list(zip(embs.embeddings, embs.vocabs, embs.vecs_names)) + list(zip(mm_embeddings, mm_vocabs, mm_labels)):
        fe, fv = filter_by_vocab(e, v, isec_vocab)
        models.append(fe)
        labels.append(l)
        np.save(os.path.join(datadir, f'{l}_common_subset'), fe)
        with open(os.path.join(datadir, f'{l}_common_subset.vocab'), 'w') as f:
            f.write('\n'.join(fv))
    # Add random embedding baseline
    models.append(np.random.random(size=(len(isec_vocab), 300)))
    labels.append('Random')

    def run(nc):
        for m, l in zip(models, labels):
            print(l)
            model_metrics, _ = run_clustering(m, cluster_method, nc, random_state, eps, min_samples, workers)
            with open(os.path.join(savedir, f'cluster_metrics_{cluster_method}_{l}_nc{nc}{suffixate(suffix)}.json'), 'w') as f:
                json.dump(model_metrics, f)

    if n_clusters == -1:
        ncs = [i * 10 for i in range(1, 9)]
        for nc in tqdm(ncs):
            run(nc)
    elif n_clusters > 0:
        run(n_clusters)


def emb_labels(fn):
    if 'model' in fn and 'resnet' in fn:
        return r'$E_L + E_V$'
    elif 'model' in fn and 'vecs3lem' in fn:
        return r'$E_L + E_S$'
    elif 'resnet' in fn and 'model' not in fn:
        return r'$E_V$'
    elif 'vecs3lem' in fn and 'model' not in fn:
        return r'$E_S$'
    elif 'model' in fn and 'resnet' not in fn and 'vecs3lem' not in fn:
        return r'$E_L$'
    elif 'Random' in fn:
        return 'Random'


def print_cluster_results(resdir='/Users/anitavero/projects/data/wikidump/models/'):
    res_files = glob(resdir + '/cluster_metrics*')
    tab = []
    header = ['Metric']
    for col, rf in enumerate(res_files):
        with open(rf, 'r') as f:
            res = json.load(f)
        header.append(emb_labels(os.path.basename(rf)))
        for row, (metric, score) in enumerate(res.items()):
            if col == 0:
                tab.append([[] for i in range(len(res_files) + 1)])
            tab[row][0] = metric
            tab[row][col + 1] = score
    table = tabulate(tab, headers=header)
    print(table)


def plot_cluster_results(resdir='/Users/anitavero/projects/data/wikidump/models/'):
    res_files = glob(resdir + '/cluster_metrics*')
    grp_files = groupby(res_files, key=lambda s: s.split('kmeans')[1].split('nc')[0])

    score_lines = defaultdict(lambda: defaultdict(list))
    ncls = []
    ncb = True
    get_num_clusters = lambda s: int(s.split('nc')[1].split('.')[0])
    for k, g in grp_files:
        nc_sorted = sorted(list(g), key=get_num_clusters)
        for fn in nc_sorted:
            if ncb:
                ncls.append(get_num_clusters(fn))
            label = emb_labels(os.path.basename(fn))
            with open(fn, 'r') as f:
                res = json.load(f)
            for metric, score in res.items():
                score_lines[metric][label].append(score)
        ncb = False

    for metric, lines in score_lines.items():
        fig, ax = plt.subplots()
        for lb, ln in lines.items():
            ax.plot(ln, label=lb, marker='o')
        ax.set_xticks(range(len(ncls)))
        ax.set_xticklabels(ncls)
        ax.set_ylabel(metric)
        ax.set_xlabel('Number of clusters')
        ax.legend(loc='best')
        plt.savefig(os.path.join(FIG_DIR, f'{metric}'), bbox_inches='tight')



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
    argh.dispatch_commands([run_clustering, run_clustering_experiments, print_cluster_results, plot_cluster_results,
                            n_nearest_neighbors, get_clustering_labels_metrics])
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