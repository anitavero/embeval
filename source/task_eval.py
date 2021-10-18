# coding: utf-8
import sys, os
import numpy as np
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from scipy.stats import spearmanr
from nltk.corpus import wordnet as wn
import spacy
from tqdm import tqdm
import json
import argh
from argh import arg
import math
import random
from typing import List, Tuple

import matplotlib
matplotlib.rcParams["savefig.dpi"] = 300
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
matplotlib.style.use('fivethirtyeight')

from itertools import combinations, product, chain
from tabulate import tabulate, LATEX_ESCAPE_RULES
from copy import deepcopy
from collections import defaultdict
import warnings

from source.process_embeddings import mid_fusion, MM_TOKEN, Embeddings
import source.utils as utils
from source.utils import get_vec, pfont, PrintFont, LaTeXFont, latex_table_post_process, latex_table_wrapper, \
                         dict2struct_array, tuple_list

sys.path.append('../2v2_software_privatefork/')
import two_vs_two

MISSING = -2  # Signify word pairs which aren't covered by and embedding's vocabulary
ROUND = 2  # Round scores in print
NAME_DELIM = ' | '
linewidth = 3


# We could use dataclass decorator in case of python 3.7
class DataSets:
    """Class for storing evaluation datasets and linguistic embeddings."""
    # Evaluation datasets
    men: List[Tuple[str, str, float]]
    simlex: List[Tuple[str, str, float]]
    # simverb: List[Tuple[str, str, float]]
    fmri_vocab = ['airplane', 'ant', 'apartment', 'arch', 'arm', 'barn', 'bear', 'bed', 'bee', 'beetle',
                  'bell', 'bicycle', 'bottle', 'butterfly', 'car', 'carrot', 'cat', 'celery', 'chair',
                  'chimney', 'chisel', 'church', 'closet', 'coat', 'corn', 'cow', 'cup', 'desk', 'dog',
                  'door', 'dress', 'dresser', 'eye', 'fly', 'foot', 'glass', 'hammer', 'hand', 'horse',
                  'house', 'igloo', 'key', 'knife', 'leg', 'lettuce', 'pants', 'pliers', 'refrigerator',
                  'saw', 'screwdriver', 'shirt', 'skirt', 'spoon', 'table', 'telephone', 'tomato', 'train',
                  'truck', 'watch', 'window']
    datasets = {}
    normalizers = {}

    def __init__(self, datadir: str):
        # SIMVERB = datadir + '/simverb-3500-data'
        # simverb_full = list(csv.reader(open(SIMVERB + '/SimVerb-3500.txt'), delimiter='\t'))
        # self.simverb = list(map(lambda x: [x[0], x[1], x[3]], simverb_full))
        self.men = json.load(open(datadir + '/men.json'))
        self.simlex = json.load(open(datadir + '/simlex.json'))
        self.datasets = {'MEN': self.men, 'SimLex': self.simlex}  # , 'SimVerb': self.simverb}
        self.normalizers = {'MEN': 50, 'SimLex': 10}  # , 'SimVerb': 10}
        self.pair_num = {'MEN': 3000, 'SimLex': 999}


def dataset_vocab(dataset: str) -> list:
    pairs = list(zip(*dataset))[:2]
    return list(set(pairs[0] + pairs[1]))


def compute_dists(vecs):
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
    for name, dataset in {'MEN': data.men, 'SimLex': data.simlex}.items():
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


def divide_eval_vocab_by_freqranges(distribution_file, eval_data_dir, dataset_name, num_groups=3, save=False):
    with open(distribution_file, 'r') as f:
        dist = json.load(f)
    ds = DataSets(eval_data_dir)
    evalv = list(set(dataset_vocab(ds.datasets[dataset_name])))

    print(f'Filter distribution file with {dataset_name} vocab')
    eval_dist = dict((w, c) for w, c in tqdm(dist.items()) if w in evalv) # Filter distribution vocab by eval vocab

    # Print and save logs
    log = f'#{dataset_name} vocab: {len(evalv)}\n'
    log += f'#Filtered Wiki vocab: {len(eval_dist)}\n'
    if len(eval_dist) < len(evalv):
       log += f'Missing words in Wiki: {list(set(evalv) - set(eval_dist.keys()))}'
    print(log)
    with open(f'{os.path.splitext(distribution_file)[0]}_{dataset_name}_split{num_groups}.log', 'w') as f:
        f.write(log)

    sorted_dist = sorted(eval_dist.items(), key=lambda item: item[1])     # sort words by frequency
    N = len(evalv)
    swords, scounts = zip(*sorted_dist)
    group_size = math.ceil(N / num_groups)
    fqvocabs = []
    for i in range(0, N, group_size):
        fmin = scounts[i]
        if N > i + group_size - 1:
            fmax = scounts[i + group_size - 1]
        else:
            fmax = scounts[-1]
        fqvocabs.append((f'{fmin} {fmax}', swords[i:i+group_size]))
    if save:    # Save vocabs for freq ranges
        with open(f'{os.path.splitext(distribution_file)[0]}_{dataset_name}_split{num_groups}.json', 'w') as f:
            json.dump(fqvocabs, f)
    return fqvocabs


def compute_correlations(scores: (np.ndarray, list), name_pairs: List[Tuple[str, str]] = None,
                         common_subset: bool = False, leave_out=False):
    """Compute correlation between score series.
        :param scores: Structured array of scores with embedding/ground_truth names.
        :param name_pairs: pairs of scores to correlate. If None, every pair will be computed.
                          if 'gt', everything will be plot against the ground_truth.
       :param leave_out: Leave out 1/leave_out portion of pairs, chosen randomly. Does not leave out if it is False.
    """
    if name_pairs == 'gt':
        name_pairs = [('ground_truth', nm) for nm in scores[0].dtype.names
                      if nm != 'ground_truth']
    elif name_pairs == 'all':
        name_pairs = None
    if not name_pairs:  # Correlations for all combinations of 2
        name_pairs = list(combinations(scores.dtype.names, 2))

    if common_subset:  # Filter rows where any of the scores are missing for a word pair
        ids = set(range(scores.shape[0]))
        for n in scores.dtype.names:
            ids = ids.intersection(set(np.where(scores[n] != MISSING)[0]))
        scs = np.array(np.empty(len(ids)), dtype=scores.dtype)
        for n in scores.dtype.names:
            scs[n] = scores[n][list(ids)]
    else:
        scs = scores

    correlations = {}
    for nm1, nm2 in name_pairs:
        # Filter pairs which the scores, coming from any of the two embeddings, don't cover
        if (scs[nm1] == MISSING).all():
            warnings.warn(f'{nm1} has 0 coverage.')
            correlations[' | '.join([nm1, nm2])] = (0, 0, 0)
        elif (scs[nm2] == MISSING).all():
            warnings.warn(f'{nm2} has 0 coverage.')
            correlations[' | '.join([nm1, nm2])] = (0, 0, 0)
        else:
            scores1, scores2 = zip(*[(s1, s2) for s1, s2 in
                                     zip(scs[nm1], scs[nm2]) if s1 != MISSING and s2 != MISSING])
            assert len(scores1) == len(scores2)
            if leave_out:
                lp = len(scores1)
                keep = 1 - 1 / leave_out
                idx = list(range(lp))
                random.shuffle(idx)
                idx = idx[:int(lp * keep)]
                scores1 = [s for i, s in enumerate(scores1) if i in idx]
                scores2 = [s for i, s in enumerate(scores2) if i in idx]
            corr = spearmanr(scores1, scores2)
            correlations[' | '.join([nm1, nm2])] = (corr.correlation, corr.pvalue, len(scores1))

    return correlations


def highlight(val, conditions: dict, tablefmt):
    """Highlight value in a table column.
    :param val: number, value
    :param conditions: dict of {colour: condition}
    :param tablefmt: 'simple' is terminal, 'latex-raw' is LaTeX
    """
    val = round(val, ROUND)
    for color, cond in conditions.items():
        if tablefmt == 'simple':
            if cond:
                return pfont([color, 'BOLD'], format(round(val, ROUND), f".{ROUND}f"), PrintFont)
        elif tablefmt in ['latex', 'latex_raw']:  # needs to be amended by hand
            if cond:
                return pfont([color, 'BOLD'], str(format(round(val, ROUND), f".{ROUND}f")), LaTeXFont)
    return format(val, f".{ROUND}f")


def mm_over_uni(name, score_dict):
    nam = deepcopy(name)
    if NAME_DELIM in nam:  # SemSim scores
        prefix, vname = nam.split(NAME_DELIM)
        prefix = prefix + NAME_DELIM
    else:  # Brain scores
        vname = nam
        prefix = ''
    if MM_TOKEN in vname:
        nm1, nm2 = vname.split(MM_TOKEN)
        get_score = lambda x: round(x, ROUND) if isinstance(x, float) else round(x[0], ROUND)
        return get_score(score_dict[name]) > get_score(score_dict[prefix + nm1]) and \
               get_score(score_dict[name]) > get_score(score_dict[prefix + nm2])
    return False


def latex_escape(string):
    return ''.join([LATEX_ESCAPE_RULES.get(c, c) for c in string])


def print_correlations(scores: np.ndarray, name_pairs='gt',
                       common_subset: bool = False, tablefmt: str = "simple", caption='', label=''):
    correlations = compute_correlations(scores, name_pairs, common_subset=common_subset)
    maxcorr = max(list(zip(*correlations.values()))[0])

    def mm_o_uni(name):
        return mm_over_uni(name, correlations)

    if 'latex' in tablefmt:
        escape = latex_escape
        font = LaTeXFont
    else:
        escape = lambda x: x
        font = PrintFont
    table = tabulate([(pfont(['ITALIC'], escape(Embeddings.get_label(nm)), font),
                       highlight(corr, {'red': corr == maxcorr, 'blue': mm_o_uni(nm)}, tablefmt),
                       format(pvalue, f".{ROUND}f"),
                       length)
                      for nm, (corr, pvalue, length) in correlations.items()],
                     headers=[pfont(['BOLD'], x, font) for x in
                              ['Name pairs', 'Spearman', 'P-value', 'Coverage']],
                     tablefmt=tablefmt)
    if 'latex' in tablefmt:
        table = latex_table_post_process(table, [3, 9], caption, label=label)

    print(table)


def print_subsampled_correlations(scores: np.ndarray, name_pairs='gt',
                       common_subset: bool = False, tablefmt: str = "simple", caption='', label='', n_sample=3):
    correlation_smpl = {}
    for i in range(n_sample):
        correlations = compute_correlations(scores, name_pairs, common_subset=common_subset, leave_out=3)
        if i == 0:
            for k, (corr, pval, len) in correlations.items():
                correlation_smpl[k] = ([corr], [pval], [len])
        else:
            for k, (corrs, pvals, lens) in correlation_smpl.items():
                corr, pval, len = correlations[k]
                correlation_smpl[k] = (corrs + [corr], pvals + [pval], lens + [len])
    maxcorr = max(list(zip(*correlations.values()))[0])

    # def mm_o_uni(name):
    #     return mm_over_uni(name, correlations)

    if 'latex' in tablefmt:
        escape = latex_escape
        font = LaTeXFont
    else:
        escape = lambda x: x
        font = PrintFont
    table = tabulate([(pfont(['ITALIC'], escape(Embeddings.get_label(nm)), font),
                       f'{format(round(np.mean(corrs), ROUND), f".{ROUND}f")} ({format(round(np.std(corrs), ROUND), f".{ROUND}f")})',
                       f'{format(round(np.mean(pvals), ROUND), f".{ROUND}f")} ({format(round(np.std(pvals), ROUND), f".{ROUND}f")})',
                       lens[0])
                      for nm, (corrs, pvals, lens) in correlation_smpl.items()],
                     headers=[pfont(['BOLD'], x, font) for x in
                              ['Name pairs', 'Spearman', 'P-value', 'Coverage']],
                     tablefmt=tablefmt)
    if 'latex' in tablefmt:
        table = latex_table_post_process(table, [3, 9], caption, label=label)

    print(table)


def print_brain_scores(brain_scores, tablefmt: str = "simple", caption='', suffix='', label=''):
    # Map for printable labels
    labels = dict((Embeddings.get_label(name), name) for name in brain_scores.keys())
    lingvnames = list(set(list(Embeddings.fasttext_vss.keys())).intersection(set(labels.keys())))
    VGvnames = [n for n in labels.keys() if n == 'VG SceneGraph']
    mm_vis_names = [n for n in labels.keys() if 'MM' in n or MM_TOKEN in n and 'VG SceneGraph' not in n]
    mm_VG_names = [n for n in labels.keys() if MM_TOKEN in n and 'VG SceneGraph' in n]
    visvnames = set(labels.keys()).difference(set(lingvnames + VGvnames + mm_VG_names + mm_vis_names))
    brain_scores_ordered = []
    # Group E_L, E_V, E_S, E_L + E_V, E_L + E_S vecs
    for n in lingvnames:
        brain_scores_ordered.append((n, brain_scores[labels[n]]))
    for n in visvnames:
        brain_scores_ordered.append((n, brain_scores[labels[n]]))
    for n in VGvnames:
        brain_scores_ordered.append((n, brain_scores[labels[n]]))
    for n in mm_vis_names:
        brain_scores_ordered.append((n, brain_scores[labels[n]]))
    for n in mm_VG_names:
        brain_scores_ordered.append((n, brain_scores[labels[n]]))

    vals = list(zip(*[v.values() for v in brain_scores.values()]))

    def print_data(data):
        # Print for individual participants and average 'fMRI' or 'MEG' scores
        v_avg = {'fMRI': 2, 'MEG': 3}[data]
        v_scores = {'fMRI': 0, 'MEG': 1}[data]
        max_avg = max(vals[v_avg])
        maxes = [max(x) for x in zip(*vals[v_scores])]
        part_num = len(list(brain_scores.values())[v_scores][data])  # number of participants
        # Scores per participant
        score_dicts = [dict((k, v[data][p]) for k, v in brain_scores_ordered) for p in range(part_num)]
        avg_dict = dict((k, v[f'{data} Avg']) for k, v in brain_scores_ordered)

        # print(f'\n-------- {data} --------\n')
        if 'latex' in tablefmt:
            escape = latex_escape
            font = LaTeXFont
        else:
            escape = lambda x: x
            font = PrintFont

        table = tabulate([[pfont(['ITALIC'], escape(name), font)] +
                          [highlight(c, {'red': c == max_scr, 'blue': mm_over_uni(name, scr_dict)}, tablefmt)
                           for c, max_scr, scr_dict in zip(v[data], maxes, score_dicts)] +
                          [highlight(v[f'{data} Avg'],
                                     {'red': v[f'{data} Avg'] == max_avg, 'blue': mm_over_uni(name, avg_dict)},
                                     tablefmt)] +
                          [round(np.std(v[data]), ROUND), v['length']]
                          for name, v in brain_scores_ordered],
                         headers=[pfont(['BOLD'], x, font) for x in
                                  ['Embedding'] +
                                  [f'P{i + 1}' for i in range(part_num)] +
                                  ['Avg', 'STD', 'Coverage']],
                         tablefmt=tablefmt)

        # Table of scores averaged for participants over all models per modality
        ling_avg_P = [np.mean([v for k, v in dictP.items() if k in lingvnames]) for dictP in score_dicts]
        vis_avg_P = [np.mean([v for k, v in dictP.items() if k in visvnames]) for dictP in score_dicts]
        VG_avg_P = [np.mean([v for k, v in dictP.items() if k in VGvnames]) for dictP in score_dicts]
        mm_vis_avg_P = [np.mean([v for k, v in dictP.items() if k in mm_vis_names]) for dictP in score_dicts]
        mm_VG_avg_P = [np.mean([v for k, v in dictP.items() if k in mm_VG_names]) for dictP in score_dicts]
        maxes = [max(p) for p in zip(ling_avg_P, vis_avg_P, VG_avg_P, mm_vis_avg_P, mm_VG_avg_P)]

        table_P = tabulate([[pfont(['ITALIC'], mod, font)] +
                            [highlight(x, {'BOLD': x == mx}, tablefmt) for x, mx in zip(avgPs, maxes)]
                            for mod, avgPs in [('$E_L$', ling_avg_P),
                                               ('$E_V$', vis_avg_P),
                                               ('$E_S$', VG_avg_P),
                                               ('$E_L + E_V$', mm_vis_avg_P),
                                               ('$E_L + E_S$', mm_VG_avg_P)]],
                           headers=[pfont(['BOLD'], x, font) for x in
                                    ['Modality'] + [f'P{i + 1}' for i in range(part_num)]],
                           tablefmt=tablefmt)
        if 'latex' in tablefmt:
            table = latex_table_post_process(table, [3, 9],
                                             f'{data} scores for each participant and embedding' + suffix + caption,
                                             fit_to_page=True, label=data + '_' + label)
            table_P = latex_table_post_process(table_P, [],
                                               f'{data} scores averaged over each modality' + suffix + \
                                               ' Bold signifies the highest average performance for each participant.',
                                               fit_to_page=True, label='_'.join([data, label, 'participants']))  # latex_table_wrapper(table_P)
        print(table, '\n')
        print(table_P)

    print_data('fMRI')
    print_data('MEG')


class PlotColour:

    @staticmethod
    def colour_by_modality(labels):
        colours = []
        linestyles = []
        alphas = []
        for l in labels:
            lst = '-'
            al = 0.8
            if MM_TOKEN in l or 'MM' in l:
                if 'VG SceneGraph' in l:
                    colours.append('#ff3e96')
                    al = 0.5
                else:
                    colours.append('#e6b830')
                    lst = ':'
                    al = 0.5
            elif l in ['wikinews', 'wikinews_sub', 'crawl', 'crawl_sub', 'w2v13']:
                colours.append('green')
            elif 'VG-' in l:
                colours.append('purple')
            elif 'VG SceneGraph' == l:
                colours.append('cyan')
            else:
                colours.append('red')
            linestyles.append(lst)
            alphas.append(al)

        return colours, linestyles, alphas

    @staticmethod
    def get_legend():
        linewidth = 3
        legends = [Line2D([0], [0], color='blue', lw=linewidth),
                   Line2D([0], [0], color='#ff3e96', lw=linewidth),
                   Line2D([0], [0], color='#e6b830', lw=linewidth),
                   Line2D([0], [0], color='green', lw=linewidth),
                   Line2D([0], [0], color='purple', lw=linewidth),
                   Line2D([0], [0], color='cyan', lw=linewidth),
                   Line2D([0], [0], color='red', lw=linewidth)]
        leglabels = ['WordNet concreteness',
                     '$E_L + E_S$',
                     '$E_L + E_V$',
                     '$E_L$',
                     '$E_{VG}$',
                     '$E_S$',
                     '$E_{Google}$']
        return legends, leglabels


def plot_brain_words(brain_scores, plot_order):
    """Plot hit counts for word in Brain data.
    :param brain_scores: brain score dict
    :param plot_order: 'concreteness' orders words for Wordnet conreteness
                       <emb_name> orders plot for an embedding's scores
    """
    vals = list(zip(*[v.values() for v in brain_scores.values()]))
    labels = Embeddings.get_labels(brain_scores.keys())

    def plot_data(ax, data, ord_vocab, ord_name):
        dscores = {'fMRI': 5, 'MEG': 6}[data]
        word_vals = vals[dscores]
        scores = {}
        wordlists = {}
        for label, val in zip(labels, word_vals):  # embeddings
            word_dict = {}  # Init dictionary with Brain vocab so we have vectors with same length to plot
            for w in ord_vocab:
                word_dict[w] = 0
            for p in val:  # participants
                for word_pair in p:  # word pairs
                    word_dict[word_pair['word1']] += word_pair['hit']
                    word_dict[word_pair['word2']] += word_pair['hit']
            # word_dict = dict(((w, word_dict[w]) for w in ord_vocab))  # Sort by concreteness
            scores[label] = list(word_dict.values())
            wordlists[label] = list(word_dict.keys())

        # Convert to structured array
        score_arrays = dict2struct_array(scores)

        tsuffix = ord_name + ' synset score'
        # Sort by ord_name Embedding
        if ord_name != 'Median' and ord_name != 'Most concrete':
            if ord_name not in labels:
                ord_name = Embeddings.get_label(ord_name)
            tsuffix = 'ordered by ' + ord_name
            score_arrays = np.sort(score_arrays, order=ord_name)
            ord_vocab = [w for w, s in sorted(zip(wordlists[ord_name], scores[ord_name]), key=lambda x: x[1])]

        colours, linestyles, alphas = PlotColour.colour_by_modality(labels)
        # allhits = sum([hits for hits in scores.values()], [])
        # ax.set_yticklabels([i for i in range(min(allhits), max(allhits), 6)], rotation=90)
        plot_scores(score_arrays,
                    vecs_names=labels,
                    labels=None,
                    colours=colours,
                    linestyles=linestyles,
                    title=f'{data} - {tsuffix}',
                    alphas=alphas,
                    xtick_labels=ord_vocab,
                    ax=ax,
                    show=False,
                    type='scatter',
                    swapaxes=True)
        ax.set_xlabel('Hit number')
        ax.yaxis.set_ticks(range(len(ord_vocab)))
        ax.set_yticklabels(ord_vocab, fontsize=14)

    if plot_order == 'concreteness':
        # Order by word concreteness
        word_concs = [[w] + list(wn_concreteness(w)) for w in DataSets.fmri_vocab]
        ord_med_vocab = [w for w, cme, cma in sorted(word_concs, key=lambda x: x[1])]
        ord_max_vocab = [w for w, cme, cma in sorted(word_concs, key=lambda x: x[2])]

        axs = [i for i in range(4)]
        fig, ((axs[0], axs[1]), (axs[2], axs[3])) = plt.subplots(2, 2, figsize=(20, 15))
        plot_data(axs[0], 'fMRI', ord_med_vocab, 'Median')
        plot_data(axs[1], 'MEG', ord_med_vocab, 'Median')
        plot_data(axs[2], 'fMRI', ord_max_vocab, 'Most concrete')
        plot_data(axs[3], 'MEG', ord_max_vocab, 'Most concrete')
    else:
        axs = [i for i in range(2)]
        fig, ((axs[0], axs[1])) = plt.subplots(1, 2, figsize=(20, 13))
        plot_data(axs[0], 'fMRI', DataSets.fmri_vocab, plot_order)
        plot_data(axs[1], 'MEG', DataSets.fmri_vocab, plot_order)

    legs, leglabels = PlotColour.get_legend()   # Leave out WordNet concreteness [1:]
    fig.legend(legs[1:], leglabels[1:], loc=9, edgecolor='inherit', ncol=7, borderaxespad=-0.2, numpoints=1, fontsize=16)
    fig.tight_layout(pad=1.0)

    return fig


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
                scores[label][i] = MISSING
            if (scores[label] == MISSING).all():
                print(f'Warning: No word pairs were found in {label} for {dataset_name}!')
        pairs.append((w1, w2))

    return scores, pairs


def plot_scores(scores: np.ndarray, gt_divisor=10, vecs_names=None, labels=None, colours=None, linestyles=None,
                title=None, type='plot', alphas=None, xtick_labels=None, ax=None, show=True, swapaxes=False):
    """Scatter plot of a structured array."""
    scs = deepcopy(scores)
    if 'ground_truth' in scores.dtype.names:
        scs['ground_truth'] /= gt_divisor

    if vecs_names is None:
        vecs_names = scs.dtype.names
    if labels is None:
        labs = [None for i in range(len(vecs_names))]
    else:
        labs = labels
    if colours is None:
        colours = [None for i in range(len(vecs_names))]
    if linestyles is None:
        linestyles = [None for i in range(len(vecs_names))]

    for nm, c, l, ls, al in zip(vecs_names, colours, labs, linestyles, alphas):
        mask = scs[nm] > MISSING  # Leave out the pairs which aren't covered
        x = np.arange(scs[nm].shape[0])[mask]
        y = scs[nm][mask]
        if swapaxes:
            buf = deepcopy(x)
            x = y
            y = buf
        if type == 'scatter':
            ax.scatter(x, y, label=l, alpha=al, color=c)
        elif type == 'plot':
            ax.plot(x, y, label=l, alpha=al, color=c, linestyle=ls, lw=linewidth)
    if labels is not None:
        ax.legend(fontsize='small', loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0.2)
    if title:
        ax.set_title(title)
    if show:
        plt.show()

    return ax


def plot_for_quantities(scores: np.ndarray, gt_divisor, common_subset=False, legend=False, pair_num=None):
    ling_names = [n for n in scores.dtype.names if 'fqrng' not in n and 'ground_truth' not in n and 'model' in n
                  and '+' not in n]
    vis_names = [n for n in scores.dtype.names if 'fqrng' not in n and 'ground_truth' not in n and 'model' not in n
                 and '+' not in n]
    mm_names = []
    for vn in vis_names:
        mm_names.append([n for n in scores.dtype.names if 'fqrng' not in n and 'ground_truth' not in n and '+' in n
                        and vn in n])
    names = ling_names + vis_names + list(chain.from_iterable(mm_names))
    quantities = sorted(list(set([int(n.split('_')[1][1:]) for n in ling_names])))
    quantities = quantities[1:] + [quantities[0]]    # -1: max train file num
    scs = scores[names + ['ground_truth']]
    scs['ground_truth'] /= gt_divisor
    correlations = compute_correlations(scs, name_pairs='gt', common_subset=common_subset)

    # Plot data with error bars
    def bar_data(nms):
        means, errs, covs = [], [], []
        for q in quantities:
            qnames = [n for n in nms if f'n{q}_' in n]
            qcorrs, qpvals, qcoverages = zip(*[correlations['ground_truth | ' + n] for n in qnames])
            q_mean, q_std, q_cov = np.mean(qcorrs), np.std(qcorrs), np.mean(qcoverages)
            means.append(q_mean)
            errs.append(q_std)
            covs.append(100 * q_cov / pair_num)   # Return coverage in percentages
        return means, errs, covs

    ling_means, ling_errs, ling_covs = bar_data(ling_names)

    def coverage_texts(xpos, means, covs, errs):
        for xp, y, cov, err in zip(xpos, means, covs, errs):
            ax.text(xp, y + err + 0.02, str(int(cov)), fontsize=20, horizontalalignment='center', rotation=90)

    fig, ax = plt.subplots()
    bar_width = 0.35
    fontsize = 20
    xpos = np.linspace(1, 2 + 2 * len(vis_names), len(quantities))

    ax.bar(np.array(xpos), ling_means, yerr=ling_errs, width=bar_width, label='$E_L$')
    if not common_subset:
        coverage_texts(xpos, ling_means, ling_covs, ling_errs)
    pi = 1
    C = 1
    # separate MM for vis_names too
    for mmn, vn in zip(mm_names, vis_names):
        mmlabel = '$E_L$ + ' + Embeddings.get_emb_type_label(vn)
        mmn_means, mmn_errs, mmn_covs = bar_data(mmn)
        mmn_xpos = np.array(xpos) + pi * bar_width
        ax.bar(mmn_xpos, mmn_means, yerr=mmn_errs, width=bar_width, label=mmlabel, color=f'C{C}')
        if not common_subset:
            coverage_texts(mmn_xpos, mmn_means, mmn_covs, mmn_errs)
        C += 1
        pi += 1
    for vn in vis_names:
        vcorr, vpval, vcoverage = correlations['ground_truth | ' + vn]
        ax.plot([xpos[0], pi * bar_width * mmn_xpos[-1]], [vcorr, vcorr],
                label=Embeddings.get_emb_type_label(vn), color=f'C{C}')
        vcoverage = 100 * vcoverage / pair_num
        if not common_subset:
            ax.text(pi * bar_width * mmn_xpos[-1] + 0.02, vcorr, str(int(vcoverage)),
                    fontsize=20)
        C += 1
        # vxpos = np.array(xpos) + pi * bar_width
        # vcorrs = [vcorr for i in xpos]
        # verrs = [0 for i in xpos]
        # ax.bar(vxpos, vcorrs, yerr=verrs, width=bar_width, label=Embeddings.get_emb_type_label(vn))
        # coverage_texts(vxpos, vcorrs, [vcoverage for i in xpos], verrs)
        # pi += 1

    ax.set_xticks(xpos)
    ax.set_xticklabels(['8M', '1G', '2G', '5G', '13G'], fontsize=fontsize)
    ax.set_ylabel('Spearman correlation', fontsize=fontsize)
    plt.tick_params(axis='both', labelsize=fontsize)
    if legend:
        ax.legend(loc=(0, 1.08), fontsize=15, ncol=5, columnspacing=1)

    if common_subset:
        return int(vcoverage)
    else:
        return False


def plot_for_freqranges(scores: np.ndarray, gt_divisor, quantity=-1, common_subset=False, pair_num=None,
                        split_num=None, ds_name=None):
    names = [n for n in scores.dtype.names if 'fqrng' in n and f'n{quantity}' in n and 'ground_truth' not in n
             and 'common_subset' not in n]
    if split_num:
        names = [n for n in names if f'split{split_num}' in n and n.count('+') <= 1]
    if ds_name:
        names = [n for n in names if ds_name in n]
    mixed = [n for n in scores.dtype.names if 'n-1' in n and 'fqrng' not in n and 'ground_truth' not in n]
    ling_names = [n for n in names if 'model' in n and '+' not in n]
    vis_names = [n for n in scores.dtype.names if 'fqrng' not in n and 'ground_truth' not in n and 'model' not in n
                 and '+' not in n]
    mm_names = []
    for vn in vis_names:
        mm_names.append([n for n in names if 'model' in n and '+' in n and vn in n])

    names = ling_names + vis_names + list(chain.from_iterable(mm_names))
    scs = scores[names + ['ground_truth'] + mixed]
    scs['ground_truth'] /= gt_divisor
    correlations = compute_correlations(scs, name_pairs='gt', common_subset=common_subset)

    freq_ranges = sorted(list(set([tuple(map(int, n.split('_')[-1].split('-'))) for n in ling_names])))

    # Plot data with error bars
    def bar_data(nms, mixed_pattern):
        means, errs, covs = [], [], []
        for fmin, fmax in freq_ranges:
            fnames = [n for n in nms if f'fqrng_{fmin}-{fmax}' in n]
            fcorrs, fpvals, fcoverages = zip(*[correlations['ground_truth | ' + n] for n in fnames])
            f_mean, f_std, f_cov = np.mean(fcorrs), np.std(fcorrs), np.mean(fcoverages)
            means.append(f_mean)
            errs.append(f_std)
            covs.append(100 * f_cov / pair_num)
        # MIXED: Full data
        mcorrs, mpvals, mcoverages = zip(*[correlations['ground_truth | ' + n] for n in mixed
                                                                        if mixed_pattern(n)])
        m_mean, m_std, m_cov = np.mean(mcorrs), np.std(mcorrs), np.mean(mcoverages)
        means.append(m_mean)
        errs.append(m_std)
        covs.append(100 * m_cov / pair_num)
        return means, errs, covs

    def coverage_texts(xpos, means, covs, errs):
        for xp, y, cov, err in zip(xpos, means, covs, errs):
            ax.text(xp, y + err + 0.02, str(int(cov)), fontsize=20, horizontalalignment='center', rotation=90)

    ling_means, ling_errs, ling_covs = bar_data(ling_names, lambda x: '+' not in x)

    fig, ax = plt.subplots()
    bar_width = 0.35
    fontsize = 20
    xpos = np.linspace(1, 2 + 2 * len(vis_names), len(freq_ranges) + 1)

    ax.bar(np.array(xpos), ling_means, yerr=ling_errs, width=bar_width, label='$E_L$')
    coverage_texts(xpos, ling_means, ling_covs, ling_errs)
    pi = 1
    C = 1
    # separate MM for vis_names too
    for mmn, vn in zip(mm_names, vis_names):
        mmlabel = '$E_L$ + ' + Embeddings.get_emb_type_label(vn)
        mmn_means, mmn_errs, mmn_covs = bar_data(mmn, lambda x: vn in x)
        mmn_xpos = np.array(xpos) + pi * bar_width
        ax.bar(mmn_xpos, mmn_means, yerr=mmn_errs, width=bar_width, label=mmlabel, color=f'C{C}')
        coverage_texts(mmn_xpos, mmn_means, mmn_covs, mmn_errs)
        pi += 1
        C += 1
    for vn in vis_names:
        vcorr, vpval, vcoverage = correlations['ground_truth | ' + vn]
        ax.plot([xpos[0], pi * bar_width * mmn_xpos[-1]], [vcorr, vcorr],
                label=Embeddings.get_emb_type_label(vn), color=f'C{C}')
        vcoverage = 100 * vcoverage / pair_num
        ax.text(pi * bar_width * mmn_xpos[-1] + 0.02, vcorr, str(int(vcoverage)),
                fontsize=20)
        C += 1
        # vxpos = np.array(xpos) + pi * bar_width
        # vcorrs = [vcorr for i in xpos]
        # verrs = [0 for i in xpos]
        # vcoverage = 100 * vcoverage / pair_num
        # ax.bar(vxpos, vcorrs, yerr=[0 for i in xpos], width=bar_width, label=Embeddings.get_label(vn))
        # coverage_texts(vxpos, vcorrs, [vcoverage for i in xpos], verrs)
        # pi += 1

    ax.set_xticks(xpos)
    ax.set_xticklabels(['LOW', 'MEDIUM', 'HIGH', 'MIXED'], fontsize=fontsize)
    ax.set_ylabel('Spearman correlation', fontsize=fontsize)
    plt.tick_params(axis='both', labelsize=fontsize)
    ax.legend(loc=(0, 1.08), fontsize=15, ncol=5, columnspacing=1)


def wn_concreteness(word, similarity_fn=wn.path_similarity):
    """WordNet distance of a word from its root hypernym."""
    syns = wn.synsets(word)
    dists = [1 - similarity_fn(s, s.root_hypernyms()[0]) for s in syns]
    return np.median(dists), max(dists)


def wn_concreteness_for_pairs(word_pairs, synset_agg: str, similarity_fn=wn.path_similarity,
                              pair_score_agg='sum') -> (np.ndarray, np.ndarray):
    """Sort scores by first and second word's concreteness scores.
    :param pair_score_agg: 'sum' adds scores for the two words, 'diff' computes their absolute difference.
    :return (ids, scores): sorted score indices and concreteness scores.
    """
    synset_agg = {'median': 0, 'most_conc': 1}[synset_agg]
    concrete_scores = [(i, wn_concreteness(w1, similarity_fn)[synset_agg],
                        wn_concreteness(w2, similarity_fn)[synset_agg])
                       for i, (w1, w2) in enumerate(word_pairs)]
    # Sort for word pairs
    if pair_score_agg == 'sum':
        agg = lambda x: x[1] + x[2]
    elif pair_score_agg == 'diff':
        agg = lambda x: np.abs(x[1] - x[2])
    ids12 = [(i, agg([i, s1, s2])) for i, s1, s2 in sorted(concrete_scores, key=agg)]
    ids, scores = list(zip(*ids12))
    return np.array(ids), np.array(scores)


def plot_by_concreteness(scores: np.ndarray, word_pairs, ax1, ax2, common_subset=False, vecs_names=None,
                         concrete_num=100, title_prefix='', pair_score_agg='sum', show=False):
    """Plot scores for data splits with increasing concreteness."""
    for synset_agg, ax in zip(['median', 'most_conc'], [ax1, ax2]):
        corrs_by_conc = defaultdict(list)
        ids12, concs = wn_concreteness_for_pairs(word_pairs, synset_agg, pair_score_agg=pair_score_agg)
        scs = scores[ids12]
        for i in range(0, len(ids12), concrete_num):
            corrs = compute_correlations(scs[i:i + concrete_num], 'gt', common_subset=common_subset)
            for k, v in corrs.items():
                corrs_by_conc[k].append(v[0])  # Append correlations score for each embedding

        corrs_by_conc_a = dict2struct_array(corrs_by_conc)

        vnames = [n for n in corrs_by_conc_a.dtype.names if 'fmri' not in n and 'frcnn' not in n]
        labels = [Embeddings.get_label(n.split(NAME_DELIM)[1]) for n in vnames]

        colours, linestyles, alphas = PlotColour.colour_by_modality(labels)
        labelpad = 10

        # Concreteness scores on different axis but the same plot
        axn = ax
        axn.plot(concs, color='blue')
        axn.set_xlabel('Word pairs', labelpad=labelpad)
        axn.set_ylabel('WordNet concreteness', labelpad=labelpad)
        axn.yaxis.label.set_color('blue')
        # Xticklabels by step size
        n = scores.shape[0]
        step = 500
        xtlabels = [i for i in range(concrete_num, n) if i % step == 0] + [n]
        axn.xaxis.set_ticks([i - 1 for i in xtlabels])
        axn.set_xticklabels(xtlabels)

        # Plot for Spearman's correlations
        axp = axn.twiny().twinx()
        axp = plot_scores(corrs_by_conc_a,
                          vecs_names=vnames,
                          labels=None,
                          colours=colours,
                          linestyles=linestyles,
                          title='',
                          alphas=alphas,
                          xtick_labels=None,
                          ax=axp,
                          show=show)
        axp.set_ylabel("Spearman's correlation", labelpad=labelpad - 3)
        # TODO: Doesn't show, order of axn.twiny().twinx() matters...
        axp.set_xlabel('WordNet concreteness splits by 100 pairs', labelpad=labelpad)
        n = corrs_by_conc_a.shape[0]
        axp.xaxis.set_ticks([i for i in range(-1, n)])
        axp.set_xticklabels(['' for i in axp.get_xticklabels()])
        syna = {'median': 'Median', 'most_conc': 'Most Concrete'}[synset_agg]
        axp.set_title(f'{title_prefix} - Synset Agg {syna}')


def eval_concreteness(scores: np.ndarray, word_pairs, num=100, gt_divisor=10, vecs_names=None, tablefmt='simple'):
    """Eval dataset instances based on WordNet synsets."""

    # Sort scores by first and second word's concreteness scores
    def print_conc(synset_agg, title):
        ids12 = wn_concreteness_for_pairs(word_pairs, synset_agg)
        # plot_scores(scores[ids1], gt_divisor, vecs_names, title=title)
        # plot_scores(scores[ids2], gt_divisor, vecs_names, title=title)
        # plot_scores(scores[ids12][:100], gt_divisor, vecs_names, title=title + ' - 100 least concrete')
        # plot_scores(scores[ids12][-100:], gt_divisor, vecs_names, title=title + ' - 100 most concrete')
        print(f'\n-------- {num} least concrete - {title} -------\n')
        print_correlations(scores[ids12][:num], name_pairs='gt', common_subset=False, tablefmt=tablefmt)
        print(f'\n-------- {num} most concrete - {title} -------\n')
        print_correlations(scores[ids12][-num:], name_pairs='gt', common_subset=False, tablefmt=tablefmt)

    # plots both for median concreteness of synsets and for the most concrete synset of words
    print_conc('median', 'Median synset concreteness')
    print_conc('most_conc', 'Most concrete synsets')


def compute_scores(actions, embeddings, scores, datasets, pairs, brain_scores=None, pre_score_files: str = None,
                   ling_vecs_names=[], vecs_names=[], mm_lingvis=False, mm_embs_of: List[Tuple[str]] = None,
                   mm_padding=False, common_subset=False):
    """Compute scores on all evaluation datasets."""
    print(actions)
    embs = embeddings.embeddings
    vocabs = embeddings.vocabs
    names = embeddings.vecs_names

    # Create multi-modal embeddings if ligvis or specific embedding pairs are given
    if mm_lingvis or mm_embs_of:
        if mm_lingvis:  # TODO: test
            mm_labels = list(product(ling_vecs_names, vecs_names))
            emb_tuples = [(embs[names.index(ln)], embs[names.index(vn)]) for ln, vn in mm_labels]
            vocab_tuples = [(vocabs[names.index(ln)], vocabs[names.index(vn)]) for ln, vn in mm_labels]
        elif mm_embs_of:  # Create MM Embeddings based on the given embedding labels
            emb_tuples = [tuple(embs[names.index(l)] for l in t) for t in mm_embs_of]
            vocab_tuples = [tuple(vocabs[names.index(l)] for l in t) for t in mm_embs_of]
            mm_labels = [tuple(l for l in t) for t in mm_embs_of]

        mm_embeddings, mm_vocabs, mm_labels = mid_fusion(emb_tuples, vocab_tuples, mm_labels, mm_padding)
        embs += mm_embeddings
        vocabs += mm_vocabs
        names += mm_labels

    if 'compscores' in actions:  # SemSim scores
        for name, dataset in datasets.datasets.items():
            dscores, dpairs = eval_dataset(dataset, name, embs, vocabs, names)
            scores[name] = dscores
            pairs[name] = dpairs

        if pre_score_files:  # Load previously saved score files and add the new scores.
            print(f'Load {pre_score_files} and join with new scores...')
            for name, dataset in datasets.datasets.items():
                pre_scores = np.load(f'{pre_score_files}_{name}.npy', allow_pickle=True)
                scores[name] = utils.join_struct_arrays([pre_scores, scores[name]])

    if 'compbrain' in actions:  # Brain scores
        if common_subset:  # Intersection of all vocabs for two_vs_two and it filters out the common subset
            vocabs = [list(set.intersection(*map(set, vocabs))) for v in vocabs]
        for emb, vocab, name in zip(embs, vocabs, names):
            fMRI_scores, MEG_scores, length, fMRI_scores_avg, MEG_scores_avg, \
            fMRI_word_scores, MEG_word_scores = two_vs_two.run_test(embedding=emb, vocab=vocab)
            brain_scores[name] = {'fMRI': fMRI_scores, 'MEG': MEG_scores,
                                  'fMRI Avg': fMRI_scores_avg, 'MEG Avg': MEG_scores_avg,
                                  'length': length,
                                  'fMRI words': fMRI_word_scores, 'MEG words': MEG_word_scores}

        if pre_score_files:  # Load previously saved score files and add the new scores.
            with open(f'{pre_score_files}_brain.json', 'r') as f:
                pre_brain_scores = json.load(f)
                for pname, pbscores in pre_brain_scores.items():
                    brain_scores[name] = pbscores

    return scores, brain_scores, pairs


# TODO: Nicer parameter handling, with exception messages
@arg('-a', '--actions', nargs='+',
     choices=['printcorr', 'plotscores', 'concreteness', 'coverage', 'compscores', 'compbrain',
              'brainwords', 'printbraincorr', 'plot_quantity', 'plot_freqrange'], default='printcorr')
@arg('-lvns', '--ling_vecs_names', nargs='+', type=str,
     choices=['w2v13', 'wikinews', 'wikinews_sub', 'crawl', 'crawl_sub'], default=[])
@arg('-vns', '--vecs_names', nargs='+', type=str)
@arg('-plto', '--plot_orders', nargs='+', type=str)
@arg('-pltv', '--plot_vecs', nargs='+', type=str)
@arg('-mmembs', '--mm_embs_of', type=tuple_list)
@arg('-pcorr', '--print_corr_for', choices=['gt', 'all'], default='all')
def main(datadir, embdir: str = None, vecs_names=[], savepath=None, loadpath=None,
         actions=['plotcorr'], plot_orders=['ground_truth'], plot_vecs=[],
         ling_vecs_names=[], pre_score_files: str = None, mm_embs_of: List[Tuple[str]] = None,
         mm_lingvis=False, mm_padding=False, print_corr_for=None, common_subset=False,
         tablefmt: str = "simple", concrete_num=100, pair_score_agg='sum', quantity=-1):
    """
    :param actions: Choose from the following:
        'printcorr': Print correlation in tables on MEN and SimLex.
        'plotscores': Plot correlations on MEN and SimLex.
        'concreteness': Scores on caption_comsub Semantic Similarity dataset splits, ordered by pair_score_agg of
                        WordNet concreteness scores of the two words in every word pair.
                        Optional: mm_padding
        'coverage': Save coverages on similarity/relatedness/brain data.
        'compscores': Compute scores on similarity/relatedness evaluation datasets.
        'compbrain': Compute scores on brain evaluation datasets.
        'brainwords': Plot Qualitative analysis on words in the brain data.
        'printbraincorr': Print correlations on brain data.
        'plot_quantity': Plot similarity/relatedness result for text quantity ranges.
        'plot_freqrange': Plot similarity/relatedness result for wor frequency ranges.
    :param pair_score_agg: 'sum' or 'diff' of concreteness scores of word pairs.
    :param mm_lingvis: if True, create multi-modal embeddings, otherwise specific embedding pairs should be given.
    :param tablefmt: printed table format. 'simple' - terminal, 'latex_raw' - latex table.
    :param concrete_num: Plot of WordNet concreteness splits by concrete_num number of pairs.
    :param datadir: Path to directory which contains evaluation data (and embedding data if embdir is not given)
    :param vecs_names: List[str] Names of embeddings
    :param embdir: Path to directory which contains embedding files.
    :param savepath: Full path to the file to save scores without extension. None if there's no saving.
    :param loadpath: Full path to the files to load scores and brain results from without extension.
                     If None, they'll be computed.
    :param plot_orders: Performance plot ordered by similarity scores of these datasets or embeddings.
    :param plot_vecs: List[str] Names of embeddings to plot scores for.
    :param ling_vecs_names: List[str] Names of linguistic embeddings.
    :param pre_score_files: Previously saved score file path without extension, which the new scores will be merged with
    :param mm_embs_of: List of str tuples, where the tuples contain names of embeddings which are to
                       be concatenated into a multi-modal mid-fusion embedding.
    :param mm_padding: Default False. Multi-modal mid-fusion method. If true, all the vectors are kept from the embeddings' vocabularies.
                       Vector representations without a vector from another modality are padded with zeros.
    :param print_corr_for: 'gt' prints correlations scores for ground truth, 'all' prints scores between all
                            pairs of scores.
    :param common_subset: action printcorr: Print results for subests of the eval datasets which are covered by all
                          embeddings' vocabularies.
                          action compbarin: Compute brain scores for interection of vocabularies.

    """

    scores = {}
    pairs = {}
    brain_scores = {}

    datasets = DataSets(datadir)

    if loadpath:
        for name, dataset in datasets.datasets.items():
            score_file = f'{loadpath}_{name}.npy'
            if os.path.exists(score_file):
                scores[name] = np.load(f'{loadpath}_{name}.npy', allow_pickle=True)
        if os.path.exists(f'{loadpath}_brain.json'):
            with open(f'{loadpath}_brain.json', 'r') as f:
                brain_scores = json.load(f)

        # For printing tables
        if 'nopadding' in loadpath:
            title_pad = 'Intersection'
            fname_pad = 'nopadding'
        else:
            title_pad = 'Padding'
            fname_pad = 'padding'

        score_explanation0 = f'Multi-modal embeddings are created using the {title_pad} technique. ' + \
                            'The table sections contain linguistic, visual and multi-modal embeddings in this order. '

        score_explanation = score_explanation0 + \
                            'Red colour signifies the best performance, blue means that the multi-modal ' + \
                            'embedding outperformed the corresponding uni-modal ones.'
    else:
        if not embdir:
            embdir = datadir
        embeddings = Embeddings(embdir, vecs_names, ling_vecs_names)

    if 'compscores' in actions or 'compbrain' in actions:
        scores, brain_scores, pairs = compute_scores(
            actions, embeddings, scores, datasets, pairs, brain_scores=brain_scores,
            pre_score_files=pre_score_files, ling_vecs_names=ling_vecs_names, vecs_names=vecs_names,
            mm_lingvis=mm_lingvis, mm_embs_of=mm_embs_of, mm_padding=mm_padding, common_subset=common_subset)

    if 'plot_quantity' in actions:
        for name in list(scores.keys()):
            scrs = deepcopy(scores[name])
            coverage = plot_for_quantities(scrs, gt_divisor=datasets.normalizers[name], common_subset=common_subset,
                                legend=True, pair_num=datasets.pair_num[name])
            cstag = f'common_subset_{coverage}_' if common_subset else ''
            plt.savefig(f'../figs/quantities_lines_{cstag}' + name + '.png', bbox_inches='tight')

    if 'plot_freqrange' in actions:
        for name in list(scores.keys()):
            scrs = deepcopy(scores[name])
            plot_for_freqranges(scrs, gt_divisor=datasets.normalizers[name], common_subset=common_subset,
                                quantity=quantity, pair_num=datasets.pair_num[name], split_num=3, ds_name=name)
            cstag = 'common_subset_' if common_subset else ''
            plt.savefig(f'../figs/eqsplit_freqranges_lines_{cstag}' + name + '.png', bbox_inches='tight')

    if 'plotscores' in actions:
        for name in list(scores.keys()):
            scrs = deepcopy(scores[name])
            for plot_order in plot_orders:  # order similarity scores of these datasets or embeddings
                plot_scores(np.sort(scrs, order=plot_order), gt_divisor=datasets.normalizers[name],
                            vecs_names=plot_vecs + ['ground_truth'])

    if 'concreteness' in actions:
        with open(loadpath + '_pairs.json', 'r') as f:
            word_pairs = json.load(f)

        if common_subset:
            commonsub = 'commonsubset'
            caption_comsub = "the embeddings' common subset of"
        else:
            commonsub = 'full'
            caption_comsub = "the full"

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        for i, (name, scrs) in enumerate(scores.items()):
            # print(f'\n-------- {name} scores -------\n')
            # eval_concreteness(scrs, word_pairs[name], num=concrete_num,
            #                   gt_divisor=datasets.normalizers[name], vecs_names=plot_vecs + ['ground_truth'],
            #                   tablefmt=tablefmt)
            plot_by_concreteness(scrs, word_pairs[name], axes[i][0], axes[i][1], common_subset=common_subset,
                                 vecs_names=plot_vecs + ['ground_truth'],
                                 concrete_num=concrete_num,
                                 title_prefix=name,
                                 pair_score_agg=pair_score_agg,
                                 show=False)

        legs, leglabels = PlotColour.get_legend()
        fig.legend(legs, leglabels, loc=9, edgecolor='inherit', ncol=7, borderaxespad=-0.2, numpoints=1)
        fig.tight_layout(pad=1.0)

        agg = {'sum': 'sum', 'diff': 'difference'}[pair_score_agg]
        fname = f'{commonsub}_{fname_pad}_{pair_score_agg}'
        fpath = 'figs/' + fname + '.png'
        latex_fig = '\\begin{figure}\n\centering\n' + \
                    '\includegraphics[width=\\textwidth]{figs/' + fname + '.png}' + \
                    '\n\caption{' + f'Scores on {caption_comsub} Semantic Similarity dataset splits, ordered by ' + \
                    f'the {agg} of WordNet concreteness scores of ' + \
                    f'the two words in every word pair. Mid-fusion method: {title_pad}.' + '}\n' + \
                    '\label{f:' + fname + '}' + \
                    '\n\end{figure}\n\n'
        # Save figure and figures tex file
        plt.savefig(fpath, bbox_inches='tight')
        with open('figs/figs_concreteness.tex', 'a+') as f:
            f.write(latex_fig)

    if 'brainwords' in actions:
        plot_brain_words(brain_scores, 'VG-VIS combined')

        if 'commonsubset' in loadpath:
            commonsub = 'commonsubset'
            caption_comsub = "the embeddings' common subset of"
        else:
            commonsub = 'full'
            caption_comsub = "the full"

        fname = f'brain_{commonsub}_{fname_pad}'
        fpath = 'figs/' + fname + '.png'
        latex_fig = '\\begin{figure}\n\centering\n' + \
                    '\includegraphics[width=\\textwidth]{figs/' + fname + '.png}' + \
                    '\n\caption{' + f'Scores on {caption_comsub} the Brain datasets words, ordered by ' + \
                    f'their WordNet concreteness. The scores are the number of hits per word, averaged over ' + \
                    f'all participants. Mid-fusion method: {title_pad}.' + '}\n' + \
                    '\label{f:' + fname + '}' + \
                    '\n\end{figure}\n\n'

        # Save figure and figures tex file
        plt.savefig(fpath, bbox_inches='tight')
        with open('figs/figs_brain.tex', 'a+') as f:
            f.write(latex_fig)

    if 'printcorr' in actions:
        if scores != {}:
            for name, scrs in scores.items():
                # print(f'\n-------- {name} scores -------\n')
                if common_subset:
                    caption = f'Spearman correlation on the common subset of the {name} dataset. '
                    commonsub = 'commonsubset'
                else:
                    caption = f'Spearman correlation on the {name} dataset. '
                    commonsub = 'full'
                print_correlations(scrs, name_pairs=print_corr_for, common_subset=common_subset,
                                   tablefmt=tablefmt, caption=caption + score_explanation,
                                   label='_'.join([name, commonsub, fname_pad]))

    if 'printcorr_subsample' in actions:
        if scores != {}:
            for name, scrs in scores.items():
                # print(f'\n-------- {name} scores -------\n')
                if common_subset:
                    caption = f'Cross-validated Spearman correlations on the common subset of the {name} dataset. '
                    commonsub = 'commonsubset'
                else:
                    caption = f'Cross-validated Spearman correlations on the {name} dataset. '
                    commonsub = 'full'
                caption += 'Spearman and P-value columns report $<$mean (STD)$>$ of three samples after leaving out the third of the evaluation pairs. '
                print_subsampled_correlations(scrs, name_pairs=print_corr_for, common_subset=common_subset,
                                   tablefmt=tablefmt, caption=caption + score_explanation0,
                                   label='_'.join([name, commonsub, fname_pad, 'crossval']), n_sample=3)

    if 'printbraincorr' in actions:
        if 'commonsubset' in loadpath:
            caption_suffix = ' on the common subset of vocabularies. '
            commonsub = 'commonsubset'
        else:
            caption_suffix = '. '
            commonsub = 'full'
        print_brain_scores(brain_scores, tablefmt=tablefmt, caption=score_explanation, suffix=caption_suffix,
                           label=f'{commonsub}_{fname_pad}')

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
    argh.dispatch_commands([main, divide_eval_vocab_by_freqranges])
