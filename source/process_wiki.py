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
from itertools import chain

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


@arg('--format', type=str, choices=['json', 'text'], default='json')
def distribution(data_dir, format='json', file_suffix=''):
    """Count word frequencies from text files or json files, containing list of str lists."""
    counter = Counter()
    files = glob(os.path.join(data_dir, '*/wiki*'))
    for fl in tqdm(files):
        with open(fl) as f:
            if format == 'text':
                tokens = f.read().split()
            elif format == 'json':
                sents = json.load(f)
                tokens = chain.from_iterable(sents)
            counter.update(tokens)
    print('Saving...')
    if file_suffix[0] != '_':
        file_suffix = '_' + file_suffix
    with open(os.path.join(data_dir, f'distribution{file_suffix}.json'), 'w') as f:
        json.dump(counter, f)


def plot_distribution(dist_file, logscale=True):
    with open(dist_file) as f:
        dist = json.load(f)
    distc = Counter(dist)
    plt.plot([v for k, v in distc.most_common()])
    if logscale:
        plt.semilogy()
    plt.show()


def create_context_files(data_dir, window=5, vocab=[], processes=1):
    files = glob(os.path.join(data_dir, '*/*.json'))
    corpus_tup = []
    for fl in files:
        with open(fl, 'r') as f:
            corpus_tup.append((fl, json.load(f)))
    text2w2vf(corpus_tup, data_dir, window=window, vocab=vocab, processes=processes)


def contexts_for_quantity(data_dir, save_dir, num, filename_suffix=''):
    """Loads randomly chosen, given number of context files and concatenates them into one file."""
    print('\nLoading contexts')
    files = glob(os.path.join(data_dir, '*/*.contexts'))
    if num > 0:
        tr_files = random.sample(files, num)
    else:   # otherwise we train on the whole corpus
        tr_files = files
    # Save training file paths
    with open(os.path.join(save_dir, f'train_files_n{num}_{filename_suffix}.txt'), 'w') as f:
        f.write('\n'.join(tr_files))
    # Read files, merge content
    contexts = ''
    for fn in tr_files:
        with open(fn) as f:
            pairs = f.read()
            if pairs[-1] != '\n':
                pairs += '\n'
            contexts += pairs
    cont_file = os.path.join(data_dir, f'n{num}_{filename_suffix}.contexts')
    with open(cont_file, 'w') as f:
        f.write(contexts)
    return cont_file


@arg('num', type=int)
def w2v_for_quantity(data_dir, save_dir, w2v_dir, num, size=300, min_count=10, workers=4,
                    negative=15, filename_suffix=''):
    """Train Word2Vec on a random number of tokenized json files.
    :param data_dir: 'tokenized' directory with subdirectories of .context files."""
    cont_file = contexts_for_quantity(data_dir, save_dir, num, filename_suffix)
    # Training Word2Vec
    print('Training')
    train_word2vecf.train(cont_file, save_dir, w2v_dir, filename_suffix=f'_n{num}_{filename_suffix}',
                          min_count=min_count, size=size, negative=negative, threads=workers)


@arg('trfile-num', type=int)
@arg('sample-num', type=int)
def w2v_for_quantities(data_dir, save_dir, w2v_dir, sample_num, trfile_num, size=300, min_count=10, workers=4,
                       negative=15, exp_name=''):
    """Train several Word2Vecs in parallel for the same data quantity, multiple times on random subsets.
    :param data_dir: 'tokenized' directory with subdirectories of jsons.
    :param save_dir: directory where we save the model and log files.
    :param sample_num: number of random trainings for the same number of files.
    :param trfile_num: number of sampled training files. If num <= 0 we train on the whole corpus.
    Rest are Word2Vec training parameters.
    """
    if exp_name:
        exp_name = '_' + exp_name

    with open(os.path.join(save_dir, f'experiment_params{exp_name}.log'), 'w') as f:
        f.write(f'Sample num: {sample_num}\n')
        f.write(f'Training file num: {trfile_num}\n')
        f.write(f'Size: {size}\n')
        f.write(f'Min count: {min_count}\n')
        f.write(f'Negative: {negative}\n')

    for i in tqdm(range(sample_num)):
        w2v_for_quantity(data_dir, save_dir, w2v_dir, trfile_num, size, min_count, workers,
                         negative, filename_suffix=f's{i}{exp_name}')


@arg('min-count', type=int)
@arg('max-count', type=int)
def contexts_for_freqrange(contexts_file, distribution_file, min_count, max_count, filename_suffix=''):
    """Filter contexts txt file for <min_count> <max_count> frequency range."""
    with open(distribution_file, 'r') as f:
        dist = json.load(f)
    print(f'Filter dictionary for frequency range {min_count}-{max_count}')
    fqvocab = map(lambda y: y[0], filter(lambda x: x[1] >= min_count and x[1] <= max_count, dist.items()))

    fqcont_file = os.path.join(os.path.split(contexts_file)[0],
                             f'freq{min_count}-{max_count}{filename_suffix}_contexts.txt')

    print('Filter contexts file')
    cf = open(contexts_file, 'r')
    fqcf = open(fqcont_file, 'w')
    with open(os.path.splitext(contexts_file)[0] + '.linenum', 'r') as f:
        n = int(f.readline())
    for i in tqdm(range(n)):
        pair = cf.readline().split()
        if pair[0] in fqvocab:
            fqcf.write(' '.join(pair) + '\n')
    fqcf.close()
    cf.close()
    return fqcont_file


@arg('min-count', type=int)
@arg('max-count', type=int)
def w2v_for_freqrange(data_dir, save_dir, w2v_dir, min_count, max_count, size=300, workers=4,
                      negative=15, exp_name=''):
    """Train Word2Vec on a corpus filtered by the given word frequency range.
    :param data_dir: 'tokenized' directory with subdirectories of .context files.
    :param save_dir: directory where we save the model and log files.
    :param min_count: minimum word frequency
    :param max_count: maximum word frequency
    Rest are Word2Vec training parameters.
    """
    if exp_name:
        exp_name = '_' + exp_name

    with open(os.path.join(save_dir, f'experiment_params{exp_name}.log'), 'w') as f:
        f.write(f'Min count: {min_count}\n')
        f.write(f'Max count: {max_count}\n')
        f.write(f'Size: {size}\n')
        f.write(f'Negative: {negative}\n')

    cont_file = contexts_for_freqrange(data_dir, save_dir, min_count, max_count, exp_name)
    # Training Word2Vec
    print('Training')
    train_word2vecf.train(cont_file, save_dir, w2v_dir,
                          filename_suffix=f'_freq{min_count}-{max_count}_{exp_name}',
                          min_count=min_count, size=size, negative=negative, threads=workers)


if __name__ == '__main__':
    argh.dispatch_commands([distribution, plot_distribution, process_files, w2v_for_quantity, w2v_for_freqrange,
                            create_context_files, contexts_for_freqrange])