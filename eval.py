# coding: utf-8
import numpy as np
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from scipy.stats import spearmanr
import csv
import spacy
from tqdm import tqdm
import json
import argh


def load_datasets(datadir):
    global men, simlex, simverb, w2v
    SIMVERB = datadir + '/simverb-3500-data'
    simverb_full = list(csv.reader(open(SIMVERB + '/SimVerb-3500.txt'), delimiter='\t'))
    simverb = list(map(lambda x: [x[0], x[1], x[3]], simverb_full))
    men = json.load(open(datadir + '/men.json'))
    simlex = json.load(open(datadir + '/simlex.json'))

    # w2v = json.load(open(datadir + '/w2v_simverb.json'))
    # w2v_simrel = json.load(open(datadir + '/simrel-wikipedia.json'))
    # w2v.update(w2v_simrel)
    # w2v = dict([(k, np.array(v)) for k, v in w2v.items()])


def dataset_vocab(dataset):
    pairs = list(zip(*dataset))[:2]
    return list(set(pairs[0] + pairs[1]))


def load_vecs(vecs_name, datadir, filter_vocab=[]):
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


def coverage(vocabulary):
    nlp = spacy.load('en')
    vocab_lemma = [[t for t in nlp(str(w))][0].lemma_ for w in vocabulary]
    vocab = set(list(vocabulary) + vocab_lemma)

    print('Vocab size:', len(vocabulary))
    print('Vocab size with lemmas:', len(vocab))

    for name, dataset in {'MEN': men, 'SimLex': simlex, 'SimVerb': simverb}.items():
        coverage = len(covered(dataset, vocabulary))
        coverage_lemma = len(covered(dataset, vocab))
        print(f'{name} pair coverage:',
              coverage_lemma, f'({round(100 * coverage_lemma / len(dataset))}%)')
        print(f'{name} pair coverage without lemmas:',
              coverage, f'({round(100 * coverage / len(dataset))}%)')


def get_vec(word):
    return vecs[np.where(vvocab == word)[0][0]].toarray().reshape(1, -1)


def eval_dataset(dataset):
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


def eval(vecs_name=None):
    if vecs_name:
        load_vecs(vecs_name)
    print('MEN')
    men_results     = [x for x in eval_dataset(men)]
    print('SimLex')
    simlex_results  = [x for x in eval_dataset(simlex)]
    print('SimVerb')
    simverb_results = [x for x in eval_dataset(simverb)]
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


def main(datadir, vecs_name=None):
    # if vecs_name == 'BOP':
    #     print("Creating BOP...")
    #     global vecs, vvocab
    #     BOP = bop.BOP('/Users/anitavero/projects/data')
    #     V = list(BOP.phondict.keys())
    #     BOP.w2bop(V)
    #     vecs = BOP.embedding
    #     vvocab = np.array(BOP.wordlist)
    # else:
    #     load_vecs(vecs_name, datadir)
    load_datasets(datadir)
    coverage(['person', 'chair', 'table', 'woman'])
    # res = eval()
    # return res

if __name__ == '__main__':
    argh.dispatch_command(main)

