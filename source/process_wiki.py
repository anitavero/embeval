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

from text_process import text2gensim
from utils import create_dir, read_jl
import embedding


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


@arg('num', type=int)
def w2v_for_quantity(data_dir, save_dir, num, size=300, window=5, min_count=10, workers=4,
                    epochs=5, max_vocab_size=None, filename_suffix=''):
    """Train Word2Vec on a random number of tokenized json files.
    :param data_dir: 'tokenized' directory with subdirectories of jsons."""
    files = glob(os.path.join(data_dir, '*/wiki*json'))
    if num > 0:     # otherwise we train on the whole corpus
        tr_files = random.sample(files, num)
    # Save training file paths
    with open(os.path.join(save_dir, f'train_files_n{num}_{filename_suffix}.txt'), 'w') as f:
        f.write('\n'.join(tr_files))
    # Read files, merge content
    corpus = []
    for fn in tr_files:
        with open(fn) as f:
            corpus += json.load(f)
    # Training Word2Vec
    embedding.train(corpus, os.path.join(save_dir, f'model_n{num}_{filename_suffix}'),
                    size=size, window=window, min_count=min_count, workers=workers,
                    epochs=epochs, max_vocab_size=max_vocab_size)


@arg('num', type=int)
@arg('sample-num', type=int)
def w2v_for_quantities(data_dir, save_dir, sample_num, num, size=300, window=5, min_count=10, workers=4,
                       epochs=5, max_vocab_size=None):
    """Train several Word2Vecs in parallel for the same data quantity, multiple times on random subsets.
    :param data_dir: 'tokenized' directory with subdirectories of jsons.
    :param save_dir: directory where we save the model and log files.
    :param sample_num: number of random trainings for the same number of files.
    :param num: number of sampled files. If num <= 0 we train on the whole corpus.
    Rest are Word2Vec training parameters.
    """
    for i in tqdm(range(sample_num)):
        w2v_for_quantity(data_dir, save_dir, num, size, window, min_count, workers,
                         epochs, max_vocab_size, filename_suffix=str(i))


def w2v_for_freqrange():
    pass


if __name__ == '__main__':
    argh.dispatch_commands([distribution, process_files, w2v_for_quantity, w2v_for_freqrange,
                            w2v_for_quantities])