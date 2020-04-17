"""Module for processing a Wikipedia dump previously extracted using
    WikiExtractor (https://github.com/attardi/wikiextractor)
"""

import os
from glob import glob
import argh
from argh import arg
import json
from tqdm import tqdm
from collections import Counter
from matplotlib import pyplot as plt
import random

from text_process import text2gensim, text2w2vf
from utils import create_dir, read_jl
import train_word2vecf

LANG = 'english'


def process_files(data_dir):
    """Sentence tokenize and stop word filter all text files
    and save the tokenized texts to json files into the 'tokenized' directory."""
    save_dir = os.path.join(data_dir, 'tokenized')
    create_dir(save_dir)
    # create the same directory structure
    dirs = glob(os.path.join(data_dir, '*'))
    dirs = list(set(dirs) - {save_dir})
    for d in dirs:
        subdir = os.path.join(save_dir, os.path.split(d)[-1])
        create_dir(subdir)

    # Tokenize files and save them
    files = glob(os.path.join(data_dir, '*/wiki*'))
    for fl in tqdm(files):
        data = read_jl(fl)
        texts = ' '.join([l['text'] for l in data])
        sent_lists = list(text2gensim(texts, LANG))
        subd, fn = fl.split('/')[-2:]
        with open(os.path.join(save_dir, subd, fn + '.json'), 'w') as f:
            json.dump(sent_lists, f)


@arg('trfile_num', type=int)
def context_pairs_for_quantity(data_dir, save_dir, trfile_num, filename_suffix=''):
    """Prepare context files for word2vecf:
        :return training_pairs:
                   textual file of word-context pairs.
                   each pair takes a separate line.
                   the format of a pair is "<word> <context>", i.e. space delimited, where <word> and <context> are strings.
                   The context is all non stop words in the same sentence.
    """
    corpus = corpus_for_quantity(data_dir, save_dir, trfile_num, filename_suffix)
    context_pairs = text2w2vf(corpus)
    with open(os.path.join(save_dir, f'context_pairs_n{trfile_num}_{filename_suffix}.txt'), 'w') as f:
        f.write(context_pairs)


def distribution(data_dir):
    """Count word frequencies from text files."""
    counter = Counter()
    files = glob(os.path.join(data_dir, '*/wiki*'))
    for fl in tqdm(files):
        with open(fl) as f:
            tokens = f.read().split()
            counter.update(tokens)
    print('Saving...')
    with open(os.path.join(data_dir, 'distribution.json'), 'w') as f:
        json.dump(counter, f)


def plot_distribution(data_dir, logscale=True):
    with open(os.path.join(data_dir, 'distribution.json')) as f:
        dist = json.load(f)
    distc = Counter(dist)
    plt.plot([v for k, v in distc.most_common()])
    if logscale:
        plt.semilogy()
    plt.show()


def corpus_for_quantity(data_dir, save_dir, num, filename_suffix=''):
    """Loads randomly chosen, given number of corpus files."""
    print('\nLoading corpus')
    files = glob(os.path.join(data_dir, '*/wiki*json'))
    if num > 0:
        tr_files = random.sample(files, num)
    else:   # otherwise we train on the whole corpus
        tr_files = files
    # Save training file paths
    with open(os.path.join(save_dir, f'train_files_n{num}_{filename_suffix}.txt'), 'w') as f:
        f.write('\n'.join(tr_files))
    # Read files, merge content
    corpus = []
    for fn in tr_files:
        with open(fn) as f:
            corpus += json.load(f)
    return corpus


@arg('num', type=int)
def w2v_for_quantity(data_dir, save_dir, w2v_dir, num, size=300, min_count=10, workers=4,
                    negative=15, filename_suffix=''):
    """Train Word2Vec on a random number of tokenized json files.
    :param data_dir: 'tokenized' directory with subdirectories of jsons."""
    corpus = corpus_for_quantity(data_dir, save_dir, num, filename_suffix)
    # Training Word2Vec
    print('Training')
    train_word2vecf.train(corpus, save_dir, w2v_dir, filename_suffix=f'_n{num}_{filename_suffix}',
                          min_count=min_count, size=size, negative=negative, threads=workers)


@arg('trfile-num', type=int)
@arg('sample-num', type=int)
def w2v_for_quantities(data_dir, save_dir, w2v_dir, sample_num, trfile_num, size=300, min_count=10, workers=4,
                       negative=15):
    """Train several Word2Vecs in parallel for the same data quantity, multiple times on random subsets.
    :param data_dir: 'tokenized' directory with subdirectories of jsons.
    :param save_dir: directory where we save the model and log files.
    :param sample_num: number of random trainings for the same number of files.
    :param trfile_num: number of sampled training files. If num <= 0 we train on the whole corpus.
    Rest are Word2Vec training parameters.
    """
    for i in tqdm(range(sample_num)):
        w2v_for_quantity(data_dir, save_dir, w2v_dir, trfile_num, size, min_count, workers,
                         negative, filename_suffix=f's{i}')


def w2v_for_freqrange():
    pass


if __name__ == '__main__':
    argh.dispatch_commands([distribution, process_files, w2v_for_quantity, w2v_for_freqrange,
                            w2v_for_quantities, context_pairs_for_quantity])