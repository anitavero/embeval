"""Module for processing a Wikipedia dump previously extracted using
    WikiExtractor (https://github.com/attardi/wikiextractor)
"""

import os
from glob import glob
import argh
from tqdm import tqdm
from collections import Counter
from text_process import tokenize


def tokenize_files(data_dir):
    """Tokenize all files and save the tokenized texts to new files into a 'tokenized' named directory."""
    save_dir = os.path.join(data_dir, 'tokenized')
    os.mkdir(save_dir)
    # create the same directory structure
    dirs = glob(os.path.join(data_dir, '*/'))
    for d in dirs:
        os.mkdir(os.path.join(save_dir, os.path.split(d)[-1]))

    # Tokenize files and save them
    files = glob(os.path.join(data_dir, '*/*'))
    for fl in tqdm(files):
        with open(fl) as f:
            tokens = tokenize(f.read())
        d, fn = os.path.split(fl)[-2:]
        with open(os.path.join(save_dir, d, fn)) as f:
            f.write(' '.join(tokens))


def distribution(data_dir):
    """Count word frequencies."""
    counter = Counter()
    files = glob(os.path.join(data_dir, '*/*'))
    for fl in tqdm(files):
        with open(fl) as f:
            tokens = f.read().split()
            counter.update(tokens)
    return counter

if __name__ == '__main__':
    argh.dispatch_command([distribution])