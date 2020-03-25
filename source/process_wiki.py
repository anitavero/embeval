"""Module for processing a Wikipedia dump previously extracted using
    WikiExtractor (https://github.com/attardi/wikiextractor)
"""

import os
import json
from glob import glob
import argh
from tqdm import tqdm
from collections import Counter
from text_process import tokenize
from utils import create_dir, read_jl


LANG = 'english'


def tokenize_files(data_dir):
    """Tokenize all json files and save the tokenized texts to text files into the 'tokenized' directory."""
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
        tokens = list(tokenize(texts, LANG))
        subd, fn = fl.split('/')[-2:]
        with open(os.path.join(save_dir, subd, fn + '.txt'), 'w') as f:
            f.write(' '.join(tokens))


def distribution(data_dir):
    """Count word frequencies from text files."""
    counter = Counter()
    files = glob(os.path.join(data_dir, '*/*'))
    for fl in tqdm(files):
        with open(fl) as f:
            tokens = f.read().split()
            counter.update(tokens)
    return counter

if __name__ == '__main__':
    argh.dispatch_commands([distribution, tokenize_files])