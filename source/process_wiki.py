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

from source.text_process import text2gensim, text2w2vf, pmi_for_words
from source.utils import create_dir, read_jl
import source.train_word2vecf as train_word2vecf

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


def get_pmi_for_words(words_file, data_dir, process=False, bigram_file=None, variant='ppmi'):
    """Save PMI scores for bigrams including words in file word_list.
        :param words_file: json file name in data_dir, consisting of an str list
        :param data_dir: path to directory with data
        :param process: bool, if True it preprocesses wiki files if False it loads preprocessed jsons.
    """
    with open(os.path.join(data_dir, words_file), 'r') as f:
        words = json.load(f)
    if process:
        process_files(data_dir)
    files = glob(os.path.join(data_dir, 'tokenized/*/wiki*json'))
    token_list = []
    for fl in tqdm(files, desc='Load files'):
        with open(fl, 'r') as f:
            token_list += list(chain.from_iterable(json.load(f)))

    pmis = pmi_for_words(words, finder_file=os.path.join(data_dir, bigram_file), token_list=token_list, variant=variant)
    print(f"Save {variant}s")
    with open(os.path.join(data_dir, words_file.replace('.', f'_WIKI_{variant}.')), 'w') as f:
        json.dump(pmis, f)


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


def create_context_files(data_dir=None, jsons=None, window=5, vocab=[], processes=1, merge=False,
                         filename_suffix=''):
    if data_dir:
        files = glob(os.path.join(data_dir, '*/*.json'))
    elif jsons:
        files = jsons
    else:
        print('Either data_dir or jsons parameter is required.')
        raise
    corpus_tup = []
    for fl in files:
        with open(fl, 'r') as f:
            corpus_tup.append((fl, json.load(f)))
    text2w2vf(corpus_tup, data_dir, window=window, vocab=vocab, processes=processes, merge=merge,
              filename_suffix=filename_suffix)


def contexts_for_quantity(data_dir, save_dir, num, filename_suffix='', contexts_pattern='',
                          window=5, vocab=[], processes=1):
    """Loads randomly chosen, given number of context files and concatenates them into one file.
        If there are no .contexts files under data_dir/* subdirectories, but one .contexts file exists under
        data_dir directly, it will just return this file name.
    """
    # Read files, merge content
    cont_file = os.path.join(data_dir, f'n{num}_{filename_suffix}.contexts')

    # If big cont_file exists, we just return it. Otherwise:
    if not os.path.exists(cont_file):
        # If no big cont_file nor .contexts files in tr_files exist,
        #  then call create_context_files with randomly sampled jsons
        print('Loading .contexts files or creating them from jsons')
        jsons = glob(os.path.join(data_dir, '*/*.json'))
        if num > 0:
            tr_jsons = random.sample(jsons, num)
        else:  # otherwise we train on the whole corpus
            tr_jsons = jsons
        # Create .contexts file from jsons only if they don't exist already.
        js2cx = lambda js: js.replace('.json', f'_window-{window}.contexts')
        tr_files = [js2cx(js) for js in tr_jsons]
        jsons2convert = [js for js in tr_jsons if not os.path.exists(js2cx(js))]
        if jsons2convert:
            print('Create contexts from:\n', '\n'.join(jsons2convert))
            create_context_files(data_dir=None, jsons=jsons2convert, window=window, vocab=vocab, processes=processes, merge=False)
        # Concatenate .contexts files into one big file
        for fn in tqdm(tr_files, desc='Concatenating contexts'):
            with open(fn) as f:
                pairs = f.read()
                if pairs and pairs[-1] != '\n':
                    pairs += '\n'
            if os.path.exists(cont_file):
                append_write = 'a'  # append if already exists
            else:
                append_write = 'w'  # make a new file if not
            with open(cont_file, append_write) as f:
                f.write(pairs)

        # Save training file paths
        with open(os.path.join(save_dir, f'train_files_n{num}_{filename_suffix}.txt'), 'w') as f:
            f.write('\n'.join(tr_files))

    return cont_file


@arg('num', type=int)
def w2v_for_quantity(data_dir, save_dir, w2v_dir, num, size=300, min_count=10, workers=4,
                    negative=15, filename_suffix='', contexts_pattern='', window=5, vocab=[]):
    """Train Word2Vec on a random number of tokenized json files.
    :param data_dir: 'tokenized' directory with subdirectories of .context files."""
    cont_file = contexts_for_quantity(data_dir, save_dir, num, filename_suffix, contexts_pattern=contexts_pattern,
                                      window=window, vocab=vocab, processes=workers)
    # Training Word2Vec
    print('Training')
    train_word2vecf.train(cont_file, save_dir, w2v_dir, filename_suffix=f'_n{num}_{filename_suffix}',
                          min_count=min_count, size=size, negative=negative, threads=workers)
    # os.remove(cont_file)    # Remove the huge concatenated context file after training


@arg('trfile-num', type=int)
@arg('sample-num', type=int)
def w2v_for_quantities(data_dir, save_dir, w2v_dir, sample_num, trfile_num, size=300, min_count=10, workers=4,
                       negative=15, exp_name='', contexts_pattern='', window=5, vocab=[]):
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
                         negative, filename_suffix=f's{i}{exp_name}', contexts_pattern=contexts_pattern,
                         window=window, vocab=vocab)


if __name__ == '__main__':
    argh.dispatch_commands([distribution, plot_distribution, process_files, w2v_for_quantity, w2v_for_quantities,
                            create_context_files, get_pmi_for_words])