# coding: utf-8
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from scipy.stats import spearmanr
import csv
import spacy
from tqdm import tqdm
import json
import argh
from argh import arg
import argparse
from typing import List, Tuple
import matplotlib.pyplot as plt
import io
from itertools import combinations, product
from tabulate import tabulate
from copy import deepcopy

from process_embeddings import mid_fusion
import utils
from utils import get_vec

sys.path.append('../2v2_software_privatefork/')
import two_vs_two


# We could use dataclass decorator in case of python 3.7
class DataSets:
    """Class for storing evaluation datasets and linguistic embeddings."""
    # Evaluation datasets
    men: List[Tuple[str, str, float]]
    simlex: List[Tuple[str, str, float]]
    simverb: List[Tuple[str, str, float]]
    fmri_vocab: List[str]
    datasets = {}
    normalizers = {}

    def __init__(self, datadir: str):
        SIMVERB = datadir + '/simverb-3500-data'
        simverb_full = list(csv.reader(open(SIMVERB + '/SimVerb-3500.txt'), delimiter='\t'))
        self.simverb = list(map(lambda x: [x[0], x[1], x[3]], simverb_full))
        self.men = json.load(open(datadir + '/men.json'))
        self.simlex = json.load(open(datadir + '/simlex.json'))
        self.fmri_vocab = ['airplane', 'ant', 'apartment', 'arch', 'arm', 'barn', 'bear', 'bed', 'bee', 'beetle', 'bell',
                          'bicycle', 'bottle', 'butterfly', 'car', 'carrot', 'cat', 'celery', 'chair', 'chimney', 'chisel',
                          'church', 'closet', 'coat', 'corn', 'cow', 'cup', 'desk', 'dog', 'door', 'dress', 'dresser',
                          'eye', 'fly', 'foot', 'glass', 'hammer', 'hand', 'horse', 'house', 'igloo', 'key', 'knife', 'leg',
                          'lettuce', 'pants', 'pliers', 'refrigerator', 'saw', 'screwdriver', 'shirt', 'skirt', 'spoon',
                          'table', 'telephone', 'tomato', 'train', 'truck', 'watch', 'window']

        self.datasets = {'MEN': self.men, 'SimLex': self.simlex, 'SimVerb': self.simverb}
        self.normalizers = {'MEN': 50, 'SimLex': 10, 'SimVerb': 10}


class Embeddings:
    """Data class for storing embeddings."""
    # Embeddings
    embeddings = List[np.ndarray]
    vocabs = List[List[str]]
    vecs_names = List[str]

    # Linguistic Embeddings
    fasttext_vss = {'wikinews': 'wiki-news-300d-1M.vec',
                    'wikinews-sub': 'wiki-news-300d-1M-subword.vec',
                    'crawl': 'crawl-300d-2M.vec',
                    'crawl-sub': 'crawl-300d-2M-subword',
                    'w2v13': ''}

    def __init__(self, datadir: str, vecs_names, ling_vecs_names=[]):
        # Load Linguistic Embeddings if they are given
        self.embeddings = []
        self.vocabs = []
        if ling_vecs_names != []:
            self.vecs_names = deepcopy(ling_vecs_names)
            for lvn in ling_vecs_names:
                if lvn == 'w2v13':
                    print(f'Loading W2V 2013...')
                    w2v = json.load(open(datadir + '/w2v_simverb.json'))
                    w2v_simrel = json.load(open(datadir + '/simrel-wikipedia.json'))
                    w2v.update(w2v_simrel)
                    self.embeddings.append(np.array(list(w2v.values())))
                    self.vocabs.append(np.array(list(w2v.keys())))
                else:
                    print(f'Loading FastText - {lvn}...')
                    fasttext_vecs, fasttext_vocab = load_fasttext(datadir + self.fasttext_vss[lvn])
                    self.embeddings.append(fasttext_vecs)
                    self.vocabs.append(fasttext_vocab)
                print('Done.')

        # Load other (visual) embeddings
        self.vecs_names += vecs_names
        for vecs_name in vecs_names:
            vecs, vocab = load_vecs(vecs_name, datadir)
            self.embeddings.append(vecs)
            self.vocabs.append(vocab)


def load_fasttext(fname: str) -> Tuple[np.ndarray, np.ndarray]:
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    fasttext_vocab = []
    fasttext_vecs = []
    for line in fin:
        tokens = line.rstrip().split(' ')
        fasttext_vocab.append(tokens[0])
        fasttext_vecs.append(list(map(float, tokens[1:])))
    return np.array(fasttext_vecs), np.array(fasttext_vocab)


def dataset_vocab(dataset: str) -> list:
    pairs = list(zip(*dataset))[:2]
    return list(set(pairs[0] + pairs[1]))


def load_vecs(vecs_name: str, datadir: str, filter_vocab=[]):
    vecs = np.load(datadir + f'/{vecs_name}.npy')
    vvocab = open(datadir + f'/{vecs_name}.vocab').read().split()
    vvocab = np.array(vvocab)
    if filter_vocab:
        vecs, vvocab = filter_by_vocab(vecs, vvocab, filter_vocab)
    return vecs, vvocab


def filter_by_vocab(vecs, vocab, filter_vocab):
    fvecs = np.empty((0, vecs[0].shape[0]))
    fvocab = []
    for w in filter_vocab:
        if w in vocab:
            fvocab.append(w)
            fvecs = np.vstack([fvecs, vecs[np.where(vocab == w)[0][0]]])
    return fvecs, fvocab


def compute_dists(vecs):
    global dists
    dists = cosine_distances(vecs, vecs)
    return dists


def neighbors(words, vocab, vecs, n=10):
    dists = compute_dists(vecs)
    if type(words[0]) == str:
        indices = [np.where(vocab == w)[0][0] for w in words]
    else:
        indices = words
    for i in indices:
        print(vocab[i], vocab[np.argsort(dists[i])[:n]])


def covered(dataset, vocab):
    return list(filter(lambda s: s[0] in vocab and s[1] in vocab, dataset))


def coverage(vocabulary, data):
    nlp = spacy.load('en')
    vvocab_lemma = [[t for t in nlp(str(w))][0].lemma_ for w in vocabulary]
    vocab = set(list(vocabulary) + vvocab_lemma)

    print('Vocab size:', len(vocabulary))
    print('Vocab size with lemmas:', len(vocab))

    # Semantic similarity/relatedness datasets
    for name, dataset in {'MEN': data.men, 'SimLex': data.simlex, 'SimVerb': data.simverb}.items():
        coverage = len(covered(dataset, vocabulary))
        coverage_lemma = len(covered(dataset, vocab))
        print(f'{name} pair coverage:',
              coverage_lemma, f'({round(100 * coverage_lemma / len(dataset))}%)')
        print(f'{name} pair coverage without lemmas:',
              coverage, f'({round(100 * coverage / len(dataset))}%)')

    # Mitchell (2008) fMRI data: 60 nouns
    coverage = len(list(set(vocabulary).intersection(set(data.fmri_vocab))))
    coverage_lemma = len(list(set(vocab).intersection(set(data.fmri_vocab))))
    print(f'fMRI coverage:', coverage_lemma, f'({round(100 * coverage_lemma / len(data.fmri_vocab))}%)')
    print(f'fMRI coverage without lemmas:', coverage, f'({round(100 * coverage / len(data.fmri_vocab))}%)')


def compute_correlations(scores: (np.ndarray, list), name_pairs: List[Tuple[str, str]] = None):
    """Computer correlation between score series.
        :param scores: Structured array of scores with embedding/ground_truth names.
        :param name_pairs: pairs of scores to correlate. If None, every pair will be computed.
    """
    if not name_pairs:
        name_pairs = list(combinations(scores.dtype.names, 2))

    correlations = {}
    for nm1, nm2 in name_pairs:
        correlations[' | '.join([nm1, nm2])] = spearmanr(scores[nm1], scores[nm2])

    return correlations


def print_correlations(scores: (np.ndarray, list), name_pairs: List[Tuple[str, str]] = None):
    correlations = compute_correlations(scores, name_pairs)
    print(tabulate([(np, corr.correlation, corr.pvalue) for np, corr in correlations.items()],
          headers=['Name pairs', 'Spearman', 'P-value']))


def eval_dataset(dataset: List[Tuple[str, str, float]],
                 dataset_name: str,
                 embeddings: List[np.ndarray],
                 vocabs: List[List[str]],
                 labels: List[str]) -> (np.ndarray, list):

    scores = np.array(np.empty(len(dataset)),
                        dtype=[('ground_truth', np.ndarray)] +
                              [(label, np.ndarray) for label in labels])
    pairs = []
    print(f'Evaluate on {dataset_name}')
    for i, (w1, w2, score) in enumerate(tqdm(dataset)):
        scores['ground_truth'][i] = float(score)
        for emb, vocab, label in zip(embeddings, vocabs, labels):
            try:
                scores[label][i] = cosine_similarity(get_vec(w1, emb, vocab), get_vec(w2, emb, vocab))[0][0]
            except IndexError:
                scores[label][i] = -2
            if (scores[label] == -2).all():
                print(f'Warning: No word pairs were found in {label} for {dataset_name}!')
        pairs.append((w1, w2))

    return scores, pairs


def plot_scores(scores: np.ndarray, gt_divisor=10, vecs_names=None) -> None:
    """Scatter plot of a structured array."""
    scs = deepcopy(scores)
    scs['ground_truth'] /= gt_divisor
    if vecs_names is None:
        vecs_names = scs.dtype.names
    for nm in vecs_names:
        mask = scs[nm] > -2   # Leave out the pairs which aren't covered
        plt.scatter(np.arange(scs[nm].shape[0])[mask], scs[nm][mask], label=nm, alpha=0.5)
    plt.legend()
    plt.show()


def qa(res, dataset='simlex'):
    scores = np.array([res[dataset]['scores'], res[dataset]['pred_scores'], res[dataset]['w2v_scores']])
    scores = scores.transpose()
    scores[:, 0] /= 10
    pairs = np.array(res[dataset]['pairs'])
    return scores, pairs


def tuple_list(arg):
    """List[Tuple[str]] argument type.
        format: whitespace separated str lists, separated by |. eg. 'embs1 embs2 | embs2 embs3 embs4'
    """
    try:
        if '|' in arg:
            tplist = [tuple(t.split()) for t in arg.split('|')]
        else:
            tplist = [tuple(arg.split())]
        return tplist
    except:
        raise argparse.ArgumentTypeError("Tuple list must be whitespace separated str lists, " +
                                         "separated by |. eg. embs1 embs2 | embs2 embs3 embs4")

# TODO: Nicer parameter handling, with exception messages
@arg('-a', '--actions', nargs='+', choices=['printcorr', 'plotscores', 'coverage', 'compscores', 'compbrain'], default='printcorr')
@arg('-lvns', '--ling_vecs_names', nargs='+', type=str, choices=['w2v13', 'wikinews', 'wikinews-sub', 'crawl', 'crawl-sub'], default='wikinews')
@arg('-vns', '--vecs_names', nargs='+', type=str)
@arg('-plto', '--plot_orders', nargs='+', type=str)
@arg('-pltv', '--plot_vecs', nargs='+', type=str)
@arg('-mmembs', '--mm_embs_of', type=tuple_list)
@arg('-pcorr', '--print_corr_for', choices=['gt', 'all'], default='all')
def main(datadir, embdir: str = None, vecs_names=[], savepath = None, loadpath = None,
         actions=['plotcorr'], plot_orders = ['ground_truth'], plot_vecs = None,
         ling_vecs_names = [], pre_score_files: str = None, mm_embs_of: List[Tuple[str]] = None,
         mm_lingvis = False, mm_padding = False, print_corr_for = None):
    """
    :param datadir: Path to directory which contains evaluation data (and embedding data if embdir is not given)
    :param vecs_names: List[str] Names of embeddings
    :param embdir: Path to directory which contains embedding files.
    :param savepath: Full path to the file to save scores without extension. None if there's no saving.
    :param loadpath: Full path to the files to load scores and brain results from without extension.
                     If None, they'll be computed.
    :param actions:
    :param gt_normalizer:
    :param plot_orders:
    :param plot_vecs:
    :param ling_vecs_names: List[str] Names of linguistic embeddings.
    :param pre_score_file: Previously saved score file path without extension, which the new scores will be merged with
    :param mm_embs_of: Choices:
                        1. List of str tuples, where the tuples contain names of embeddings which are to
                           be concatenated into a multi-modal mid-fusion embedding.
                        2. ling_vis: compines all given vecs_names and ling_vecs_names
    :param mm_padding:
    :param print_corr_for: 'gt' prints correlations scores for ground truth, 'all' prints scores between all
                            pairs of scores.
    """

    scores = {}
    pairs = {}
    brain_scores = {}

    datasets = DataSets(datadir)

    if loadpath:
        for name, dataset in datasets.datasets.items():
            scores[name] = np.load(f'{loadpath}_{name}.npy', allow_pickle=True)
        with open(f'{loadpath}_brain.json', 'r') as f:
            brain_scores = json.load(f)
    else:
        if not embdir:
            embdir = datadir
        embeddings = Embeddings(embdir, vecs_names, ling_vecs_names)

    if 'compscores' in actions or 'compbrain' in actions:
        print(actions)
        embs = embeddings.embeddings
        vocabs = embeddings.vocabs
        names = embeddings.vecs_names

        if mm_lingvis:    # TODO: test
            mm_labels = list(product(ling_vecs_names, vecs_names))
            emb_tuples = [(embs[names.index(ln)], embs[names.index(vn)]) for ln, vn in mm_labels]
            vocab_tuples = [(vocabs[names.index(ln)], vocabs[names.index(vn)]) for ln, vn in mm_labels]
        elif mm_embs_of:    # Create MM Embeddings based on the given embedding labels
            emb_tuples = [tuple(embs[names.index(l)] for l in t) for t in mm_embs_of]
            vocab_tuples = [tuple(vocabs[names.index(l)] for l in t) for t in mm_embs_of]
            mm_labels = [tuple(l for l in t) for t in mm_embs_of]

        mm_embeddings, mm_vocabs, mm_labels = mid_fusion(emb_tuples, vocab_tuples, mm_labels, mm_padding)
        embs += mm_embeddings
        vocabs += mm_vocabs
        names += mm_labels

        if 'compscores' in actions: # SemSim scores
            for name, dataset in datasets.datasets.items():
                dscores, dpairs = eval_dataset(dataset, name, embs, vocabs, names)
                scores[name] = dscores
                pairs[name] = dpairs

            if pre_score_files:   # Load previously saved score files and add the new scores.
                print(f'Load {pre_score_files} and join with new scores...')
                for name, dataset in datasets.datasets.items():
                    pre_scores = np.load(f'{pre_score_files}_{name}.npy', allow_pickle=True)
                    scores[name] = utils.join_struct_arrays([pre_scores, scores[name]])

        if 'compbrain' in actions:  # Brain scores
            for emb, vocab, name in zip(embs, vocabs, names):
                fMRI_score, MEG_score, length = two_vs_two.run_test(embedding=emb, vocab=vocab)
                brain_scores[name] = {'fMRI': fMRI_score, 'MEG': MEG_score, 'lenght': length}

            if pre_score_files:  # Load previously saved score files and add the new scores.
                with open(f'{pre_score_files}_brain.json', 'r') as f:
                    pre_brain_scores = json.load(f)
                    for pname, pbscores in pre_brain_scores.items():
                        brain_scores[name] = pbscores

    if 'plotscores' in actions:
        for name in list(scores.keys()):
            scrs = deepcopy(scores[name])
            for plot_order in plot_orders:  # order similarity scores of these datasets or embeddings
                plot_scores(np.sort(scrs, order=plot_order), gt_divisor=datasets.normalizers[name],
                            vecs_names=plot_vecs + ['ground_truth'])

    if 'printcorr' in actions:
        if scores is not None:
            if print_corr_for == 'gt':
                name_pairs = [('ground_truth', nm) for nm in list(scores.values())[0].dtype.names]
            elif print_corr_for == 'all':
                name_pairs = None   # Will print score correlations for all combinations of 2
            for name, scrs in scores.items():
                print(f'\n-------- {name} scores -------\n')
                print_correlations(scrs, name_pairs=name_pairs)

        print('\n-------- Brain scores -------\n')
        print(tabulate([(name, v['fMRI'], v['MEG'], v['lenght']) for name, v in brain_scores.items()],
                       headers=['Embedding', 'fMRI avg', 'MEG avg', '#Vocab of 60']))

    if 'coverage' in actions:
        for vocab, name in zip(embeddings.vocabs, embeddings.vecs_names):
            print('\n--------------' + name + '--------------\n')
            coverage(vocab, datasets)

    if savepath:
        print('Saving...')
        if scores != {}:
            for name, scs in scores.items():
                np.save(savepath + f'_{name}.npy', scs)
            with open(savepath + '_pairs.json', 'w') as f:
                json.dump(pairs, f)
        if brain_scores != {}:
            with open(savepath + '_brain.json', 'w') as f:
                json.dump(brain_scores, f)


if __name__ == '__main__':
    argh.dispatch_command(main)
