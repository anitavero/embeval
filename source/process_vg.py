import os
import json
from collections import Counter
import argh
import spacy
from tqdm import tqdm

from source.text_process import pmi_for_words


def vg_dists(datadir='/Users/anitavero/projects/data/visualgenome'):
    with open(os.path.join(datadir, 'vg_contexts_rad3_lemmatised1.txt'), 'r') as f:
        ctx = [pair.split() for pair in tqdm(f.read().split('\n'))]
    words = []
    for pair in ctx:
        if len(pair) < 2:
            print('MISSING', pair)
        else:
            words.append(pair[0])
    with open(os.path.join(datadir, 'vg_contexts_rad3_lemmatised1_dists.json'), 'w') as f:
        json.dump(Counter(words), f)


def vg_pmis(words_file, datadir='/Users/anitavero/projects/data/visualgenome'):
    """Save PMI scores for bigrams including words in file word_list.
        :param words_file: json file name in data_dir, consisting of an str list
        :param datadir: path to directory with data
    """
    with open(os.path.join(datadir, words_file), 'r') as f:
        words = json.load(f)
    with open(os.path.join(datadir, 'vg_contexts_rad3_lemmatised1.txt'), 'r') as f:
        ctx = [pair.split() for pair in tqdm(f.read().split('\n'), desc='Read VG contexts')]
    pmis = pmi_for_words(words, document_list=ctx)
    print("Save PMIs")
    with open(os.path.join(datadir, words_file.replace('.', '_VG_pmi.')), 'w') as f:
        json.dump(pmis, f)


def description_corpus(region_descriptions, lemmatise):
    """Return all descriptions as a corpus in form of list of strings (sentences)."""
    nlp = spacy.load('en')
    corpus = []
    for rg in tqdm(region_descriptions):
        if lemmatise:
            corpus += [' '.join([w.lemma_ for w in nlp(r['phrase']) if not w.is_punct])
                       for r in rg['regions']]
        else:
            corpus += [r['phrase'] for r in rg['regions']]
    return corpus


def save_description_corpus(datadir, lemmatise=True):
    region_descriptions = json.load(open(datadir + '/region_descriptions.json'))
    corpus = description_corpus(region_descriptions, lemmatise)
    with open(datadir + '/region_description_corpus.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(corpus))


if __name__ == '__main__':
    argh.dispatch_commands([vg_dists, vg_pmis, save_description_corpus])