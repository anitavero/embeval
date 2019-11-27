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
import copy
import io
from itertools import combinations

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

    # Linguistic Embeddings
    w2v_vecs: np.ndarray
    w2v_vocab: List[str]
    fasttext_vecs: np.ndarray
    fasttext_vocab: List[str]


    def __init__(self, datadir: str, ling: bool=True):
        SIMVERB = datadir + '/simverb-3500-data'
        simverb_full = list(csv.reader(open(SIMVERB + '/SimVerb-3500.txt'), delimiter='\t'))
        self.simverb = list(map(lambda x: [x[0], x[1], x[3]], simverb_full))
        self.men = json.load(open(datadir + '/men.json'))
        self.simlex = json.load(open(datadir + '/simlex.json'))

        if ling:    # only load linguistic embeddings if ling==True
            w2v = json.load(open(datadir + '/w2v_simverb.json'))
            w2v_simrel = json.load(open(datadir + '/simrel-wikipedia.json'))
            w2v.update(w2v_simrel)
            self.w2v_vecs = np.array(list(w2v.values()))
            self.w2v_vocab = np.array(list(w2v.keys()))

            print('Loading FastText...')
            self.fasttext_vecs, self.fasttext_vocab = load_fasttext(datadir + '/wiki-news-300d-1M.vec')
            print('Done.')

        self.fmri_vocab = ['airplane', 'ant', 'apartment', 'arch', 'arm', 'barn', 'bear', 'bed', 'bee', 'beetle', 'bell',
                          'bicycle', 'bottle', 'butterfly', 'car', 'carrot', 'cat', 'celery', 'chair', 'chimney', 'chisel',
                          'church', 'closet', 'coat', 'corn', 'cow', 'cup', 'desk', 'dog', 'door', 'dress', 'dresser',
                          'eye', 'fly', 'foot', 'glass', 'hammer', 'hand', 'horse', 'house', 'igloo', 'key', 'knife', 'leg',
                          'lettuce', 'pants', 'pliers', 'refrigerator', 'saw', 'screwdriver', 'shirt', 'skirt', 'spoon',
                          'table', 'telephone', 'tomato', 'train', 'truck', 'watch', 'window']

        self.datasets = {'MEN': self.men, 'SimLex': self.simlex, 'SimVerb': self.simverb}


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
        correlations['-'.join([nm1, nm2])] = spearmanr(scores[nm1], scores[nm2])

    return correlations


def print_correlations(scores: (np.ndarray, list), name_pairs: List[Tuple[str, str]] = None):
    correlations = compute_correlations(scores, name_pairs)
    print('Spearman correlations:')
    for np, corr in correlations.items():
        print(f'{np}: {corr.correlation} (p={corr.pvalue})')


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
            if (scores[label] == 2).all():
                print('Warning: No word pairs were found!')
        pairs.append((w1, w2))

    return scores, pairs


def plot_scores(scores: np.ndarray, gt_divisor=10) -> None:
    """Scatter plot of a structured array."""
    scs = copy.deepcopy(scores)
    scs['ground_truth'] /= gt_divisor
    for nm in scs.dtype.names:
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


@arg('-a', '--actions', nargs='+', choices=['printcorr', 'plotscores', 'coverage', 'compscores', 'compbrain'], default='printcorr')
@argh.arg('-vns', '--vecs_names', nargs='+', type=str)
@argh.arg('-plto', '--plot_orders', nargs='+', type=str)
@argh.arg('-mmembs', '--mm_embs_of', type=tuple_list)
def main(datadir, vecs_names=[], vecsdir: str = None, savepath = None, loadpath = None,
         actions=['plotcorr'], gt_normalizer = 10, plot_orders = ['ground_truth'], ling = False,
         pre_score_file: str = None, mm_embs_of: List[Tuple[str]] = None, mm_padding = False):
    """
    :param datadir:
    :param vecs_names:
    :param vecsdir:
    :param savepath: Full path to the file to save scores without extension. None if there's no saving.
    :param loadpath: Full path to the files to load scores and brain results from without extension.
                     If None, they'll be computed.
    :param actions:
    :param gt_normalizer:
    :param plot_orders:
    :param ling: True if we load linguistic embeddings.
    :param pre_score_file: Previously saved score file path, which the new scores will be merged with
    :param mm_embs_of: List of str tuples, where the tuples contain names of embeddings which are to
                       be concatenated into a multi-modal mid-fusion embedding.
    """

    scores = None
    brain_scores = None
    if not loadpath:
        if not vecsdir:
            vecsdir = datadir + '/mmdeed'

        vis_embeddings = []
        vis_vocabs = []
        for vecs_name in vecs_names:
            vecs, vocab = load_vecs(vecs_name, vecsdir)
            vis_embeddings.append(vecs)
            vis_vocabs.append(vocab)

        data = DataSets(datadir, ling)

    else:
       scores = np.load(loadpath + '.npy', allow_pickle=True)
       with open(loadpath + '_brain.json', 'r') as f:
           brain_scores = json.load(f)

    if 'compscores' in actions or 'compbrain' in actions:
        print(actions)
        embs = vis_embeddings
        vocabs = vis_vocabs
        names = vecs_names

        if ling:
            embs += [data.w2v_vecs, data.fasttext_vecs]
            vocabs += [data.w2v_vocab, data.fasttext_vocab]
            names += ['w2v', 'fasttext']

        if mm_embs_of:  # Create MM Embeddings based on the given embedding labels
            emb_tuples = [tuple(embs[names.index(l)] for l in t) for t in mm_embs_of]
            vocab_tuples = [tuple(vocabs[names.index(l)] for l in t) for t in mm_embs_of]
            mm_labels = [tuple(l for l in t) for t in mm_embs_of]
            mm_embeddings, mm_vocabs, mm_labels = mid_fusion(emb_tuples, vocab_tuples, mm_labels, mm_padding)
            embs += mm_embeddings
            vocabs += mm_vocabs
            names += mm_labels

        if 'compbrain' not in actions:
            scores, pairs = eval_dataset(data.datasets['MEN'], 'MEN', embs, vocabs, names)

            if pre_score_file:   # Load previously saves score file and add the new scores.
                print(f'Load {pre_score_file} and join with new scores...')
                pre_scores = np.load(pre_score_file, allow_pickle=True)
                scores = utils.join_struct_arrays([pre_scores, scores])

        # Brain scores
        brain_scores = {}
        for emb, vocab, name in zip(embs, vocabs, names):
            fMRI_score, MEG_score, length = two_vs_two.run_test(embedding=emb, vocab=vocab)
            brain_scores[name] = {'fMRI': fMRI_score, 'MEG': MEG_score, 'lenght': length}

    if 'plotscores' in actions:
        for plot_order in plot_orders:  # order similarity scores of these datasets or embeddings
            plot_scores(np.sort(scores, order=plot_order), gt_divisor=gt_normalizer)

    if 'printcorr' in actions:
        if scores is not None:
            print_correlations(scores)
        print('\n-------- Brain scores -------\n')
        for name in brain_scores.keys():
            print(name)
            print("There are %d words found from the input" % brain_scores[name]['lenght'])
            print("The fMRI avg score is %f" % brain_scores[name]['fMRI'])
            print("The MEG avg score is %f" % brain_scores[name]['MEG'])

    if 'coverage' in actions:
        for vocab, name in zip([data.w2v_vocab, data.fasttext_vocab] + vis_vocabs,
                               ['w2v', 'fasttext'] + vecs_names):
            print('\n--------------' + name + '--------------\n')
            coverage(vocab, data)

    if savepath:
        print('Saving...')
        np.save(savepath + '.npy', scores)
        with open(savepath + '_pairs.json', 'w') as f:
            json.dump(pairs, f)
        with open(savepath + '_brain.json', 'w') as f:
            json.dump(brain_scores, f)


if __name__ == '__main__':
    argh.dispatch_command(main)
