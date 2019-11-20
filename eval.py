# coding: utf-8
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from scipy.stats import spearmanr
import csv
import spacy
from tqdm import tqdm
import json
import argh
from collections import defaultdict
from typing import List, Tuple
import matplotlib.pyplot as plt
import copy



def load_datasets(datadir: str) -> None:
    global men, simlex, simverb, w2v
    SIMVERB = datadir + '/simverb-3500-data'
    simverb_full = list(csv.reader(open(SIMVERB + '/SimVerb-3500.txt'), delimiter='\t'))
    simverb = list(map(lambda x: [x[0], x[1], x[3]], simverb_full))
    men = json.load(open(datadir + '/men.json'))
    simlex = json.load(open(datadir + '/simlex.json'))

    w2v = json.load(open(datadir + '/w2v_simverb.json'))
    w2v_simrel = json.load(open(datadir + '/simrel-wikipedia.json'))
    w2v.update(w2v_simrel)
    w2v = dict([(k, np.array(v)) for k, v in w2v.items()])


def dataset_vocab(dataset: str) -> list:
    pairs = list(zip(*dataset))[:2]
    return list(set(pairs[0] + pairs[1]))


def load_vecs(vecs_name: str, datadir: str, filter_vocab=[]):
    global vecs, vvocab
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


dists = []
def compute_dists():
    global dists
    dists = cosine_distances(vecs, vecs)


def neighbors(words, n=10):
    if type(words[0]) == str:
        indices = [np.where(vvocab == w)[0][0] for w in words]
    else:
        indices = words
    for i in indices:
        print(vvocab[i], vvocab[np.argsort(dists[i])[:n]])


def covered(dataset, vocab):
    return list(filter(lambda s: s[0] in vocab and s[1] in vocab, dataset))


def coverage(vocabulary=None):
    if not vocabulary:
        vocabulary = vvocab
    nlp = spacy.load('en')
    vvocab_lemma = [[t for t in nlp(str(w))][0].lemma_ for w in vocabulary]
    vocab = set(list(vocabulary) + vvocab_lemma)

    print('Vocab size:', len(vocabulary))
    print('Vocab size with lemmas:', len(vocab))

    for name, dataset in {'MEN': men, 'SimLex': simlex, 'SimVerb': simverb}.items():
        coverage = len(covered(dataset, vocabulary))
        coverage_lemma = len(covered(dataset, vocab))
        print(f'{name} pair coverage:',
              coverage_lemma, f'({round(100 * coverage_lemma / len(dataset))}%)')
        print(f'{name} pair coverage without lemmas:',
              coverage, f'({round(100 * coverage / len(dataset))}%)')


def get_vec(word, embeddings=vecs, vocab=vvocab):
    return embeddings[np.where(vocab == word)[0][0]].reshape(1, -1)



def eval_vg_dataset(dataset):
    scores = []
    pred_scores = []
    w2v_scores = []
    pairs = []
    for w1, w2, score in tqdm(dataset):
        if w1 in vvocab and w2 in vvocab:
            scores.append(float(score))
            pred_scores.append(cosine_similarity(get_vec(w1), get_vec(w2))[0][0])
            w2v_scores.append(cosine_similarity(w2v[w1].reshape(1, -1), w2v[w2].reshape(1, -1))[0][0])
            pairs.append((w1, w2))
    vg_spearman = spearmanr(scores, pred_scores)
    w2v_spearman = spearmanr(scores, w2v_scores)
    print(f'\nVG Spearman: {vg_spearman.correlation} (p={vg_spearman.pvalue})')
    print(f'w2v Spearman: {w2v_spearman.correlation} (p={w2v_spearman.pvalue})')
    return vg_spearman, w2v_spearman, scores, pred_scores, w2v_scores, pairs


def eval_dataset(dataset: List[Tuple[str, str, float]],
                 embeddings: List[np.ndarray]=[w2v, vecs],
                 vocabs: List[List[str]]=[vvocab, vvocab],
                 labels: List[str]=['w2v', 'vc']) -> (np.ndarray, list):

    scores = np.array(np.empty(len(dataset)),
                        dtype=[('ground_truth', np.ndarray)] +
                              [(label, np.ndarray) for label in labels])
    pairs = []
    for i, (w1, w2, score) in enumerate(tqdm(dataset)):
        scores['ground_truth'][i] = float(score)
        for emb, vocab, label in zip(embeddings, vocabs, labels):
            try:
                scores[label][i] = cosine_similarity(get_vec(w1, emb, vocab), get_vec(w2, emb, vocab))[0][0]
            except:
                scores[label][i] = -2
        # w2v_scores.append(cosine_similarity(w2v[w1].reshape(1, -1), w2v[w2].reshape(1, -1))[0][0])
        pairs.append((w1, w2))

    return scores, pairs


def plot_scores(scores: np.ndarray) -> None:
    '''Scatter plot of a sstructured array.'''
    scs = copy.deepcopy(scores)
    scs['ground_truth'] /= 10
    for nm in scs.dtype.names:
        mask = scs[nm] > -2   # Leave out the pairs which aren't covered
        plt.scatter(np.arange(scs[nm].shape[0])[mask], scs[nm][mask])
    plt.show()


def eval(vecs_name=None):
    if vecs_name:
        load_vecs(vecs_name)
    print('MEN')
    men_results     = [x for x in eval_vg_dataset(men)]
    print('SimLex')
    simlex_results  = [x for x in eval_vg_dataset(simlex)]
    print('SimVerb')
    simverb_results = [x for x in eval_vg_dataset(simverb)]
    res_names = ['vg_spearman', 'w2v_spearman', 'scores', 'pred_scores', 'w2v_scores', 'pairs']

    results = {'men':     dict(zip(res_names, men_results)),
               'simlex':  dict(zip(res_names, simlex_results)),
               'simverb': dict(zip(res_names, simverb_results))}
    return results


def qa(res, dataset='simlex'):
    scores = np.array([res[dataset]['scores'], res[dataset]['pred_scores'], res[dataset]['w2v_scores']])
    scores = scores.transpose()
    scores[:, 0] /= 10
    pairs = np.array(res[dataset]['pairs'])
    return scores, pairs


def main(datadir, vecs_name, vecsdir=None, save=False, savedir=None):
    if not vecsdir:
        vecsdir = datadir
    load_vecs(vecs_name, vecsdir)
    load_datasets(datadir)
    # coverage()
    res = eval_dataset()

    if save:
        if not savedir:
            savedir = datadir
        with open(os.path.join(savedir, vecs_name +'_res.json'), 'w') as f:
            json.dump(res, f)

#    return res

if __name__ == '__main__':
    argh.dispatch_command(main)

