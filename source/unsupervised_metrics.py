import os
import argh
from argh import arg
import re
import numpy as np
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import cosine
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
import seaborn as sns
from itertools import groupby
from collections import defaultdict, Counter, OrderedDict
from tabulate import tabulate, LATEX_ESCAPE_RULES
import pandas as pd

from source.utils import suffixate, tuple_list, pfont, latex_table_post_process, PrintFont, LaTeXFont
from source.process_embeddings import Embeddings, mid_fusion, filter_by_vocab

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figs')


################# Nearest Neigbor metrics #################


def get_n_nearest_neighbors(words: np.ndarray, E: np.ndarray, vocab: np.ndarray, n: int = 10):
    """n nearest neighbors for words based on cosine distance in Embedding E."""
    w_idx = np.where(np.in1d(vocab, np.array(words)))[0]  # Words indices in Vocab and Embedding E
    C = cosine_distances(E)
    np.fill_diagonal(C, np.inf)
    w_C = C[:, w_idx]  # Filter columns for words
    nNN = np.argpartition(w_C, range(n), axis=0)[:n]  # Every column j contains the indices of NNs of word_j
    return np.vstack([words, vocab[nNN]])  # 1st row: words, rows 1...n: nearest neighbors


@arg('-w', '--words', nargs='+', type=str)
def n_nearest_neighbors(data_dir, model_name, words=[], n: int = 10):
    """n nearest neighbors for words based on model <vecs_names>."""
    embs = Embeddings(data_dir, [model_name])
    E, vocab = embs.embeddings[0], embs.vocabs[0]
    return get_n_nearest_neighbors(np.array(words), E, vocab, n).transpose()


################# Clusterization metrics #################


def cluster_method_from_filename(fn):
    if 'kmeans' in fn:
        method = 'kmeans'
    elif 'agglomerative' in fn:
        method = 'agglomerative'
    return method


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
    centroids = kmeans_model.cluster_centers_
    return labels, centroids


def agglomerative_clustering(model, n_clusters=3, linkage='ward'):
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=n_clusters).fit(model)
    labels = clustering.labels_
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


def run_clustering(model, cluster_method, n_clusters=3, random_state=1, eps=0.5, min_samples=5,
                   workers=4, linkage='ward'):
    if type(model) == str:
        model = np.load(model)

    centroids = []
    if cluster_method == 'dbscan':
        labels = dbscan_clustering(model, eps=eps, min_samples=min_samples, n_jobs=workers)
    elif cluster_method == 'kmeans':
        labels, centroids = kmeans(model, n_clusters=n_clusters, random_state=random_state, n_jobs=workers)
    elif cluster_method == 'agglomerative':
        labels = agglomerative_clustering(model, n_clusters=n_clusters, linkage=linkage)

    return cluster_eval(model, labels), labels, centroids


@arg('-vns', '--vecs_names', nargs='+', type=str, required=True)
def get_clustering_labels_metrics(vecs_names=[], datadir='/anfs/bigdisc/alv34/wikidump/extracted/models/',
                                  savedir='/anfs/bigdisc/alv34/wikidump/extracted/models/results/',
                                  cluster_method='kmeans', n_clusters=3, random_state=1, eps=0.5, min_samples=90,
                                  workers=4, suffix='', linkage='ward'):
    embs = Embeddings(datadir, vecs_names)
    for e, v, l in list(zip(embs.embeddings, embs.vocabs, embs.vecs_names)):
        print(l)
        model_metrics, cl_labels, centroids = run_clustering(e, cluster_method, n_clusters, random_state, eps,
                                                             min_samples, workers, linkage)
        with open(os.path.join(savedir,
                               f'cluster_metrics_labelled_{cluster_method}_{l}_nc{n_clusters}{suffixate(suffix)}.json'),
                  'w') as f:
            json.dump(model_metrics, f)

        label_dict = {w: str(l) for l, w in zip(cl_labels, v)}
        cluster_label_filepath = \
            os.path.join(savedir, f'cluster_labels_{cluster_method}_{l}_nc{n_clusters}{suffixate(suffix)}.json')
        with open(cluster_label_filepath, 'w') as f:
            json.dump(label_dict, f)

        inspect_clusters(cluster_label_filepath)  # Save printable clusters dict

        if centroids != []:
            # Save distances from centroids
            dists = distances_from_centroids(e, v, label_dict, centroids)
            with open(os.path.join(savedir,
                                   f'dists_from_centr_{cluster_method}_{l}_nc{n_clusters}{suffixate(suffix)}.json'),
                      'w') as f:
                json.dump(dists, f)


def distances_from_centroids(emb, vocab, label_dict, centroids):
    dists = {}
    V = np.array(vocab)
    for w, cl in label_dict.items():
        i = np.where(V == w)[0][0]
        dists[w] = cosine(emb[i, :], centroids[int(cl), :])
    return dists


def order_words_by_centroid_distance(clusters, cluster_label_filepath):
    """Order words by their distance from the centroid"""
    path, fn = os.path.split(cluster_label_filepath)
    dist_file = os.path.join(path, '_'.join(['dists_from_centr'] + fn.split('_')[2:]))
    with open(dist_file, 'r') as f:
        cent_dists = json.load(f)

    for cl, words in clusters:
        words.sort(key=lambda w: cent_dists[w])


def synset_closures(word, depth=3, get_names=False):
    hyper = lambda s: s.hypernyms()
    synsets = wn.synsets(word)
    closures = list(chain.from_iterable([list(sns.closure(hyper, depth=depth)) for sns in synsets])) + synsets
    if get_names:
        closures = [str(syns).split('.')[0].split("'")[1] for syns in closures]
    return closures


def wn_label_for_words(words, depth=3):
    sys_name_list = []
    for w in words:
        sys_name_list += list(set(synset_closures(w, depth=depth, get_names=True)))
    label = Counter(sys_name_list)
    return [w for w, fq in label.most_common()]


def label_clusters_with_wordnet(depth=3, max_label_num=3):
    """First max_label_num most common synset names."""
    with open('cluster_files.json', 'r') as f:
        cf = json.load(f)
    for clfile in cf['clfiles']:
        with open(os.path.join(cf['datapath'], clfile), 'r') as f:
            cls = json.load(f)

        wncls = []
        for clid, words in cls:
            wncls.append((clid, wn_label_for_words(words, depth)[:max_label_num], words))

        with open(os.path.join(cf['datapath'], clfile.replace('clusters', 'clusters_WN')), 'w') as f:
            json.dump(wncls, f)


def inspect_clusters(cluster_label_filepath):
    """
    Convert cluster label file containing {word: label} dict to {cluster_id: wordlist} dict,
     ordered by the number of cluster members.
    :param cluster_label_filepath:
    :return:
    """
    with open(cluster_label_filepath, 'r') as f:
        cl_dict = json.load(f)

    clusters = defaultdict(list)
    for w, c in cl_dict.items():
        clusters[int(c)].append(w)

    clusters = sorted([(cl, ws) for cl, ws in clusters.items()], key=lambda x: len(x[1]))

    # Save clusters
    with open(cluster_label_filepath.replace('cluster_labels', 'clusters'), 'w') as f:
        json.dump(clusters, f)


def print_clusters(clusters_WN_filepath, tablefmt, barfontsize=20):
    """

    :param clusters_WN_filepath:
    :param tablefmt: printed table format. 'simple' - terminal, 'latex_raw' - latex table.
    :param barfontsize:
    :return:
    """
    # Table of cluster members ordered by size
    embtype = emb_labels(os.path.split(clusters_WN_filepath)[-1])

    method = cluster_method_from_filename(clusters_WN_filepath)

    with open(clusters_WN_filepath, 'r') as f:
        clusters = json.load(f)

    cluster_num = len(clusters)
    if 'latex' in tablefmt:
        font = LaTeXFont
        labelform = lambda ls: '\\makecell[tr]{' + '\\\\'.join(ls).replace('_', ' ') + '}'
    else:
        font = PrintFont
        labelform = lambda x: x

    table = tabulate([(labelform(wnls), cl, ', '.join(words)) for cl, wnls, words in clusters],
                     headers=[pfont(['BOLD'], x, font) for x in ['WN label', 'Own label', 'Members']],
                     tablefmt=tablefmt)

    if 'latex' in tablefmt:
        table = latex_table_post_process(table, range(0, cluster_num - 1),
                                         f'Members of the {len(clusters)} clusters in {embtype}. Clusters are ordered by size.',
                                         label=f'{embtype}_{len(clusters)}_clusters')
        with open(f'figs/{embtype}_{len(clusters)}_clusters_{method}.tex', 'w') as f:
            f.write(table)

    avgfreq_file = clusters_WN_filepath.replace('WN', 'avgfeq')
    if os.path.exists(avgfreq_file):
        with open(avgfreq_file, 'r') as f:
            cl_freqs = json.load(f)
    else:
        cl_freqs = None

    cluster_sizes_avgfreq(clusters, cl_freqs, embtype, method, barfontsize)

    return clusters, table


def cluster_sizes_avgfreq(clusters, cl_freqs, embtype=None, method=None, barfontsize=20, suffix=''):
    """Historgram of cluster sizes"""
    cluster_num = len(clusters)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = '#30a2da'
    ax1.set_xlabel('Clusters', fontsize=barfontsize)
    ax1.set_ylabel('#Members', fontsize=barfontsize, color=color)
    ax1.bar(range(cluster_num), [len(ws) for cl, wnls, ws in clusters], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(range(cluster_num))
    ax1.set_xticklabels(np.arange(cluster_num) + 1)
    ax1.semilogy()

    if cl_freqs:
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = '#fc4f30'
        ax2.set_ylabel('Avg word frequency', fontsize=barfontsize, color=color)  # we already handled the x-label with ax1
        ax2.plot([mf for i, c, mf, sf in cl_freqs], color=color)
        ax2.tick_params(axis='y', labelcolor=color)

    plt.tight_layout()
    plt.savefig(f'figs/{embtype}_{len(clusters)}_cluster_hist_{method}{suffixate(suffix)}.png')


def jaccard_similarity_score(x, y):
    """
    Jaccard Similarity J (A,B) = | Intersection (A,B) | /
                                    | Union (A,B) |
    """
    intersection_cardinality = len(set(x).intersection(set(y)))
    union_cardinality = len(set(x).union(set(y)))
    return intersection_cardinality / float(union_cardinality)


def order_clusters_by_avgfreq(clusters, datapath, clfile):
    with open(os.path.join(datapath, clfile.replace('WN', 'avgfeq')), 'r') as f:
        cl_freqs = json.load(f)
    cl_freqs = sorted(cl_freqs, key=lambda x: x[2])
    clusters = {clid: [wn_label, words] for clid, wn_label, words in clusters}
    ordered_cls = []
    for clid, words, fq_mean, fq_std in cl_freqs:
        ordered_cls.append([clid] + clusters[clid] + [fq_mean])

    return ordered_cls


def cluster_similarities(order='default', clmethod='agglomerative', plot=True):
    datapath = '/Users/anitavero/projects/data/wikidump/models/results'
    emb_clusters1, emb_clusters2 = {}, {}

    def read_clusters(clmethod):
        emb_clusters = OrderedDict()
        clfiles = [f'clusters_WN_{clmethod}_vecs3lem1_common_subset_nc20.json',
                   f'clusters_WN_{clmethod}_model_n-1_s0_window-5_common_subset_nc20.json']
        if order != 'avgfreq':
            clfiles.append(f'clusters_WN_{clmethod}_google_resnet152_common_subset_nc20.json')

        for clfile in clfiles:
            with open(os.path.join(datapath, clfile), 'r') as f:
                clusters = json.load(f)
            embtype = emb_labels(os.path.split(clfile)[-1])
            emb_clusters[embtype] = clusters
            if order == 'avgfreq':
                emb_clusters[embtype] = order_clusters_by_avgfreq(clusters, datapath, clfile)

        return emb_clusters

    if clmethod == 'kmeans' or clmethod == 'agglomerative':
        emb_clusters = read_clusters(clmethod)
        emb_clusters1, emb_clusters2 = emb_clusters, emb_clusters
        compare = 'cross'
    elif clmethod == 'kmeans_agglomerative':
        emb_clusters1 = read_clusters('kmeans')
        emb_clusters2 = read_clusters('agglomerative')
        compare = 'dot'

    return compute_cluster_similarities(emb_clusters1, emb_clusters2, compare, order, clmethod, plot)


def compute_cluster_similarities(emb_clusters1, emb_clusters2, compare, order, clmethod, plot):
    """

    :param emb_clusters1:
    :param emb_clusters2:
    :param compare: 'cross' or 'dot'
    :param order:
    :param clmethod:
    :param plot:
    :return:
    """
    def compute_sim(e, e1, cls, cls1):
        sims = np.empty((20, 20))
        xticks, yticks = [], []
        for i, c in enumerate(cls):
            yticks.append(', '.join(c[1]) + (f' {round(c[3], 5)}' if order == 'avgfreq' else ''))
            for j, c1 in enumerate(cls1):
                if len(xticks) < 20:
                    xticks.append(', '.join(c1[1]) + (f' {round(c1[3], 5)}' if order == 'avgfreq' else ''))
                sims[i, j] = jaccard_similarity_score(c[2], c1[2])
        jaccard_similarities[f'{e}-{e1}'] = sims

        if plot:
            if order == 'clustermap':
                similarity_clustermap(sims, xticks, yticks, f'{e}-{e1}_{clmethod}')
            elif order == 'default' or order == 'avgfreq':
                similarity_heatmap(sims, xticks, yticks, f'{e}-{e1}_{clmethod}', order)
            else:
                pass

    jaccard_similarities = {}
    if compare == 'cross':
        for ie, (e, cls) in enumerate(emb_clusters1.items()):
            for ie1, (e1, cls1) in enumerate(emb_clusters2.items()):
                if ie < ie1:
                    compute_sim(e, e1, cls, cls1)
    elif compare == 'dot':
        for (e, cls), (e1, cls1) in zip(emb_clusters1.items(), emb_clusters2.items()):
            compute_sim(e, e1, cls, cls1)

    return jaccard_similarities


def similar_cluster_nums(clmethod='agglomerative'):
    jss = cluster_similarities(clmethod=clmethod, plot=False)
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    nums = defaultdict(list)
    maxs = dict()
    for k in jss.keys():
        for th in thresholds:
            nums[k].append(np.count_nonzero(jss[k] > th))
        maxs[k] = np.max(jss[k])

    return nums, maxs


def similarity_clustermap(V, xticks, yticks, title_embs):
    df = pd.DataFrame(V, columns=xticks, index=yticks)
    plt.subplots(figsize=(20, 16))
    cm = sns.clustermap(df, linewidths=.1)

    # plt.setp(cm.ax_heatmap.xaxis.get_majorticklabels(), rotation=45)  # Messes up the labels...
    cm.fig.suptitle(f"Jaccard similarities of clusters in {title_embs}")
    cm.savefig(f'figs/{title_embs}_jaccard_clustermap.png')


def similarity_heatmap(V, xticks, yticks, title_embs, order):
    fig, ax = plt.subplots(figsize=(20, 16))
    ax = sns.heatmap(V, linewidths=.1)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(xticks)))
    ax.set_yticks(np.arange(len(yticks)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(xticks, fontsize=20)
    ax.set_yticklabels(yticks, fontsize=20)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")

    ax.set_title(f"Jaccard similarities of clusters in {title_embs}")
    fig.tight_layout()
    plt.savefig(f'figs/{title_embs}_jaccard_heatmap_{order}.png')


def run_print_clusters(barfontsize=25):
    with open('cluster_files.json', 'r') as f:
        cf = json.load(f)
    for clfile in cf['clfiles']:
        print_clusters(os.path.join(cf['datapath'], clfile), 'latex_raw', barfontsize=barfontsize)


def run_inspect_clusters():
    datapath = '/Users/anitavero/projects/data/wikidump/models/results'
    for clfile in ['cluster_labels_kmeans_vecs3lem1_common_subset_nc20.json',
                   'cluster_labels_kmeans_vecs3lem1_common_subset_nc40.json',
                   'cluster_labels_kmeans_model_n-1_s0_window-5_common_subset_nc20.json',
                   'cluster_labels_kmeans_google_resnet152_common_subset_nc20.json']:
        inspect_clusters(os.path.join(datapath, clfile))


def avg_cluster_wordfrequency(datadir='/Users/anitavero/projects/data/', clmethod='agglomerative'):
    with open(os.path.join(datadir, 'wikidump/tokenized/common_subset_vocab_VG_GoogleResnet_Wiki2020.json'), 'r') as f:
        vocab = json.load(f)

    with open(os.path.join(datadir, 'wikidump/tokenized/distribution.json'), 'r') as f:
        wiki_dist = json.load(f)
    wiki_dist_com = {w: c for w, c in tqdm(wiki_dist.items(), desc='Wiki') if w in vocab}

    with open(os.path.join(datadir, 'visualgenome/vg_contexts_rad3_lemmatised1_dists.json'), 'r') as f:
        vg_dist = json.load(f)
    vg_dist_com = {w: c for w, c in tqdm(vg_dist.items(), desc='VG rel') if w in vocab}

    assert len(wiki_dist_com) == len(vg_dist_com)

    # Relative frequencies
    sum_wiki = sum(wiki_dist_com.values())
    wiki_dist_com = {w: 100 * c / sum_wiki for w, c in wiki_dist_com.items()}
    wiki_dist_com = {k: v for k, v in sorted(wiki_dist_com.items(), key=lambda item: item[1])}

    sum_vg = sum(vg_dist_com.values())
    vg_dist_com = {w: 100 * c / sum_vg for w, c in vg_dist_com.items()}
    vg_dist_com = {k: v for k, v in sorted(vg_dist_com.items(), key=lambda item: item[1])}

    freqs = {'wiki': wiki_dist_com, 'vg': vg_dist_com}

    # Avg rel freqs for clusters
    datapath = '/Users/anitavero/projects/data/wikidump/models/results'
    for clfile, mod in [(f'clusters_{clmethod}_vecs3lem1_common_subset_nc20.json', 'vg'),
                        (f'clusters_{clmethod}_model_n-1_s0_window-5_common_subset_nc20.json', 'wiki')]:
        with open(os.path.join(datapath, clfile), 'r') as f:
            cls = json.load(f)
        fqcls = []
        for clid, words in cls:
            cl_freqs = [freqs[mod][w] for w in words]
            fqcls.append((clid, words, np.mean(cl_freqs), np.std(cl_freqs)))

        with open(os.path.join(datapath, clfile.replace('clusters', 'clusters_avgfeq')), 'w') as f:
            json.dump(fqcls, f)


def vg_dists(datadir='/Users/anitavero/projects/data/visualgenome'):
    with open(os.path.join(datadir, 'vg_contexts_rad3_lemmatised1.txt'), 'r') as f:
        ctx = [pair.split() for pair in tqdm(f.read().split('\n'))]
    words = []
    for pair in ctx:
        if len(pair) < 2:
            print('MISSING', pair)
        else:
            words.append(pair[0])
    with open(os.path.join(datadir, 'vg_contexts_rad3_lemmatised1_dists.json'), 'w') as f:
        json.dump(Counter(words), f)


@arg('-mmembs', '--mm_embs_of', type=tuple_list)
@arg('-vns', '--vecs_names', nargs='+', type=str, required=True)
def run_clustering_experiments(datadir='/anfs/bigdisc/alv34/wikidump/extracted/models/',
                               savedir='/anfs/bigdisc/alv34/wikidump/extracted/models/results/',
                               vecs_names=[], mm_embs_of=[], cluster_method='dbscan', n_clusters=-1, random_state=1,
                               eps=0.5, min_samples=90, workers=4, suffix='', linkage='ward'):
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
    for e, v, l in list(zip(embs.embeddings, embs.vocabs, embs.vecs_names)) + list(
            zip(mm_embeddings, mm_vocabs, mm_labels)):
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
            model_metrics, _, _ = run_clustering(m, cluster_method, nc, random_state, eps, min_samples, workers,
                                                 linkage)
            with open(os.path.join(savedir, f'cluster_metrics_{cluster_method}_{l}_nc{nc}{suffixate(suffix)}.json'),
                      'w') as f:
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
                            n_nearest_neighbors, get_clustering_labels_metrics, inspect_clusters, run_inspect_clusters,
                            label_clusters_with_wordnet, run_print_clusters, cluster_similarities,
                            avg_cluster_wordfrequency, vg_dists, similar_cluster_nums])
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
