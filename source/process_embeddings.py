import os
import pickle
import json
import numpy as np
from tqdm import tqdm
from itertools import combinations
from typing import List, Tuple
from gensim.models import Word2Vec
import re
import io
from copy import deepcopy
from glob import glob
import argparse, argh
from argh import arg

from source.utils import get_file_name


class Embeddings:
    """Data class for storing embeddings."""
    # Embeddings
    embeddings = List[np.ndarray]
    vocabs = List[List[str]]
    vecs_names = List[str]
    vecs_labels = List[str]

    # Linguistic Embeddings
    fasttext_vss = {'wikinews': 'wiki-news-300d-1M.vec',
                    'wikinews_sub': 'wiki-news-300d-1M-subword.vec',
                    'crawl': 'crawl-300d-2M.vec',
                    'crawl_sub': 'crawl-300d-2M-subword',
                    'w2v13': ''}

    def __init__(self, datadir: str, vecs_names, ling_vecs_names=None):
        # Load Linguistic Embeddings if they are given
        if ling_vecs_names is None:
            ling_vecs_names = []
        self.embeddings = []
        self.vocabs = []
        self.vecs_names = []
        if ling_vecs_names:
            self.vecs_names = deepcopy(ling_vecs_names)
            for lvn in ling_vecs_names:
                if lvn == 'w2v13':
                    print(f'Loading W2V 2013...')
                    w2v = json.load(open(datadir + '/w2v_simverb.json'))
                    w2v_simrel = json.load(open(datadir + '/simrel-wikipedia.json'))
                    w2v.update(w2v_simrel)
                    self.embeddings.append(np.array(list(w2v.values())))
                    self.vocabs.append(np.array(list(w2v.keys())))
                else:
                    print(f'Loading FastText - {lvn}...')
                    fasttext_vecs, fasttext_vocab = self.load_fasttext(datadir + self.fasttext_vss[lvn])
                    self.embeddings.append(fasttext_vecs)
                    self.vocabs.append(fasttext_vocab)
                print('Done.')

        # Load other (visual) embeddings
        self.vecs_names += vecs_names
        for vecs_name in vecs_names:
            vecs, vocab = self.load_vecs(vecs_name, datadir)
            self.embeddings.append(vecs)
            self.vocabs.append(vocab)

        self.vecs_labels = [self.get_label(name) for name in self.vecs_names]

    @staticmethod
    def get_labels(name_list):
        return [Embeddings.get_label(name) for name in name_list]

    @staticmethod
    def get_label(name):
        """Return a printable label for embedding names."""
        name = re.sub('ground_truth [-|\|] ', '', name)  # Remove ground_truth prefix

        def label(nm):
            cnn_format = {'vgg': 'VGG', 'alexnetfc7': 'AlexNet', 'alexnet': 'AlexNet',
                          'resnet-18': 'ResNet-18', 'resnet152': 'ResNet-152'}
            mod_format = {'vs': 'VIS', 'mm': 'MM'}
            if 'frcnn' in nm:
                _, context, modality, _ = nm.split('_')
                return f'Google-{mod_format[modality]} {context}'
            elif 'fmri' in nm:
                if 'combined' in nm:
                    _, context, modality, _ = nm.split('_')
                    return f'VG-{mod_format[modality]} {context}'
                elif 'descriptors' in nm:
                    context, _, modality, _ = nm.split('-')[1].split('_')
                    return f'VG-{mod_format[modality]} {context}'
                else:
                    _, data, cnn = nm.split('_')
                    return f'{data.capitalize()} {cnn_format[cnn]}'
            elif 'men' in nm:
                _, context = nm.split('-')
                return f'VG-{context}'
            elif 'vecs' in nm:
                return 'VG SceneGraph'
            elif 'model' in nm:
                return nm
            elif nm not in Embeddings.fasttext_vss.keys():
                data, cnn = nm.split('_')
                return f'{data.capitalize()} {cnn_format[cnn]}'
            else:
                return nm

        if MM_TOKEN in name:
            name1, name2 = name.split(MM_TOKEN)
            return label(name1) + MM_TOKEN + label(name2)
        else:
            return label(name)

    def load_fasttext(self, fname: str) -> Tuple[np.ndarray, np.ndarray]:
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        fasttext_vocab = []
        fasttext_vecs = []
        for line in fin:
            tokens = line.rstrip().split(' ')
            fasttext_vocab.append(tokens[0])
            fasttext_vecs.append(list(map(float, tokens[1:])))
        return np.array(fasttext_vecs), np.array(fasttext_vocab)

    def load_vecs(self, vecs_name: str, datadir: str, filter_vocab=[]):
        """Load npy vector files and vocab files. If they are not present load try loading gensim model."""
        path = datadir + f'/{vecs_name}'
        try:
            if os.path.exists(path + '.vocab'):
                vecs = np.load(path + '.npy')
                vvocab = open(path + '.vocab').read().split()
                vvocab = np.array(vvocab)
            else:
                model = Word2Vec.load(path)
                vecs = model.wv.vectors
                vvocab = np.array(list(model.wv.vocab.keys()))
        except FileNotFoundError as err:
            print(f'{err.filename} not found.')
            return
        if filter_vocab:
            vecs, vvocab = filter_by_vocab(vecs, vvocab, filter_vocab)
        return vecs, vvocab


def serialize2npy(filepath: str, savedir: str, maxnum: int = 10):
    """Save embedding files from pickle containing dictionary of {word: np.ndarray}
        into embedding.npy, embedding.vocab, for eval.
        The embedding is a numpy array of shape(vocab size, vector dim)
        Vocabulary is a text file including words separated by new line.
        :param filepath: Path to a pickle file containing a dict of
                either {word: <image embedding list>}
                or     {word: <image embedding>}        ('descriptors' suffix in mmfeat file names)
    """
    filename, ext = os.path.basename(filepath).split('.')

    if ext == 'pkl':
        with open(filepath, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')  # Load python2 pickles
    elif ext == 'json':
        with open(filepath, 'r') as f:
            data_dict = json.load(f)

    # Save vocabulary
    with open(os.path.join(savedir, filename + '.vocab'), 'w') as f:
        try:    # TODO: review handling str
            vocab = [str(s, 'utf-8') for s in data_dict.keys()]
        except:
            vocab = [str(s) for s in data_dict.keys()]
        f.write('\n'.join(vocab))

    values = list(data_dict.values())
    if isinstance(values[0], dict):
        print(f'Aggregating max {maxnum} number of image representations for each word...')
        embeddings = agg_img_embeddings(values, maxnum)
    elif isinstance(values[0], np.ndarray):
        embeddings = np.array(values)

    # Save embedding
    np.save(os.path.join(savedir, filename + '.npy'), embeddings)


def agg_img_embeddings(values: dict, maxnum: int = 10) -> np.ndarray:
    """Aggregate image vectors from a dictionary of to numpy embeddings and vocabulary.
        The embedding is a numpy array of shape(vocab size, vector dim)
        Vocabulary is a text file including words separated by new line.
    """
    # Aggregate image vectors for a word, using the fist min(maxnum, imangenum) images
    embeddings = np.empty((len(values), np.array(list(values[0].values())).shape[1]))
    for i, imgs in enumerate(tqdm(values)):
        vecs = np.array(list(imgs.values()))
        embeddings[i] = vecs[:min(maxnum, vecs.shape[0])].mean(axis=0)
    return embeddings


MM_TOKEN = '+'  # Connects embedding names for labelling, e.g. 'linguistic+visual'

def mid_fusion(embeddings, vocabs, labels,
               padding: bool, combnum: int = 2) -> (List[np.ndarray], List[np.ndarray], List[str]):
    """Concatenate embeddings pairwise for words in the intersection or union (with padding) of their vocabulary.
        :param embeddings: List[np.ndarray] or List[Tuple[np.ndarray]]
        :param vocabs: List[np.ndarray] or List[Tuple[np.ndarray]]
        :param labels: List[np.ndarray] or List[Tuple[np.ndarray]]
        :param padding: If true, all the vectors are kept from the embeddings' vocabularies.
                        The vectors parts without a vector from another modality are padded with zeros.
        :param combnum: number of modalities concatenated in the final multi-modal vector
    """
    # TODO: generalise to MM embeddings containing more than 2 modalities
    if isinstance(embeddings[0], np.ndarray):
        id_pairs = list(combinations(range(len(embeddings)), combnum))
        emb_pairs = []
        vocab_pairs = []
        label_pairs = []
        for id1, id2 in id_pairs:
            emb_pairs.append((embeddings[id1], embeddings[id2]))
            vocab_pairs.append((vocabs[id1], vocabs[id2]))
            label_pairs.append((labels[id1], labels[id2]))
    if isinstance(embeddings[0], tuple):
        emb_pairs = embeddings
        vocab_pairs = vocabs
        label_pairs = labels

    mm_embeddings = []
    mm_vocabs = []
    mm_labels = []
    for (emb1, emb2), (vocab1, vocab2), (label1, label2) in zip(emb_pairs, vocab_pairs, label_pairs):
        shape1 = emb1.shape[1]
        shape2 = emb2.shape[1]
        label = MM_TOKEN.join([label1, label2])
        if padding:
            print(f'MM {label} with padding:')
            mm_vocab = list(set(vocab1).union(set(vocab2)))
            mm_embedding = np.zeros((len(mm_vocab), shape1 + shape2))

            print('Creating index...')
            idx = {x: i for i, x in enumerate(mm_vocab)}
            idx1 = [idx[w] for w in vocab1]
            idx2 = [idx[w] for w in vocab2]

            print('Creating MM Embeddings...')
            mm_embedding[idx1, :shape1] = emb1
            mm_embedding[idx2, shape1:] = emb2
        else:
            print(f'MM {label} without padding:')
            mm_vocab = list(set(vocab1).intersection(set(vocab2)))
            mm_embedding = np.zeros((len(mm_vocab), shape1 + shape2))

            print('Creating index...')
            idx = {x: i for i, x in enumerate(mm_vocab)}
            idx_v1 = {x: i for i, x in enumerate(vocab1)}
            idx_v2 = {x: i for i, x in enumerate(vocab2)}
            idx1, idx_emb1 = zip(*[(idx[w], idx_v1[w]) for w in vocab1 if w in mm_vocab])
            idx2, idx_emb2 = zip(*[(idx[w], idx_v2[w]) for w in vocab2 if w in mm_vocab])

            print('Creating MM Embeddings...')
            mm_embedding[idx1, :shape1] = emb1[idx_emb1, :]
            mm_embedding[idx2, shape1:] = emb2[idx_emb2, :]

        mm_embeddings.append(mm_embedding)
        mm_vocabs.append(np.array(mm_vocab))
        mm_labels.append(label)

        assert mm_embedding.shape == (len(mm_vocab), emb1.shape[1] + emb2.shape[1])

    return mm_embeddings, mm_vocabs, mm_labels


def filter_by_vocab(vecs, vocab, filter_vocab):
    """Filter numpy array and corresponding vocab, so they contain words and vectors for
        words in filter_vocab."""
    if filter_vocab == []:
        return [], []
    vidx = {x: i for i, x in tqdm(enumerate(vocab), desc='Vocab index')}
    print('Computing intersection of vocab and filter vocab.')
    intersect = set(vocab).intersection(set(filter_vocab))
    idx = sorted([vidx[w] for w in tqdm(intersect, desc='Emb index and filtered vocab')])
    print('Filter embedding and vocab')
    fvocab = vocab[idx]
    fvecs = vecs[np.array(idx, dtype=int), :]
    return fvecs, list(fvocab)


def filter_for_freqranges(datadir, file_pattern, distribution_file, num_groups=3):
    """Filter embedding files with the given file pattern.
        :param num_groups: int, number of frequency groups. The groups have approximately equal frequency mass.
    """
    print(f'Divide vocab to {num_groups} splits with approx. equal mass')
    fqvocabs = divide_vocab_by_freqranges(distribution_file, num_groups)

    vecs_names = [get_file_name(path) for path in glob(os.path.join(datadir, f'*{file_pattern}*.npy'))]
    print('Load embeddings')
    embs = Embeddings(datadir, vecs_names)
    fembs = {}
    print('Filter embeddings for freq ranges')
    for emb, vocab, label in zip(embs.embeddings, embs.vocabs, embs.vecs_labels):
        for fqrange, fqvocab in fqvocabs.items():
            fmin, fmax = fqrange.split()
            print(f'{label}, Freq: {fmin} - {fmax}')
            # TODO      Parallelize.
            # TODO      Filter for all freq range in one filter_by_vocab
            # TODO      Filter by eval task vocab too?
            femb, fvocab = filter_by_vocab(emb, vocab, fqvocab)
            fembs[f'{fmin} {fmax}'] = {'label': label, 'vecs': femb, 'vocab': fvocab}

            # Save embeddings and vocabs for freq range
            new_label = f'{datadir}/{label}_fqrng_{fmin}-{fmax}'
            with open(f'{new_label}.vocab', 'w') as f:
                f.write('\n'.join(fvocab))
            np.save(f'{new_label}.npy', femb)

    return fembs


def divide_vocab_by_freqranges(distribution_file, num_groups=3, save=False):
    with open(distribution_file, 'r') as f:
        dist = json.load(f)
    sorted_dist = sorted(dist.items(), key=lambda item: item[1])    # sort words by frequency
    sum_mass = sum(dist.values())
    group_mass = sum_mass // num_groups
    fqvocabs = {}
    group_sum = 0
    fqvocab = []
    fmin = sorted_dist[0][1]
    vocablen = len(sorted_dist)
    for i in tqdm(range(vocablen)):
        w, c = sorted_dist[i]
        fqvocab.append(w)
        group_sum += c
        if group_sum > group_mass:
            fqvocabs[f'{fmin} {sorted_dist[i-1][1]}'] = fqvocab[:-1]
            if save:
                # Save embeddings and vocabs for freq range
                new_label = f'{os.path.splitext(distribution_file)[0]}_fqrng_{fmin}-{sorted_dist[i-1][1]}'
                with open(f'{new_label}.vocab', 'w') as f:
                    f.write('\n'.join(fqvocab))
            fqvocab = [w]
            fmin = c
            group_sum = c
        if i == vocablen - 1:
            fqvocabs[f'{fmin} {sorted_dist[i][1]}'] = fqvocab
            if save:
                new_label = f'{os.path.splitext(distribution_file)[0]}_fqrng_{fmin}-{sorted_dist[i][1]}'
                with open(f'{new_label}.vocab', 'w') as f:
                    f.write('\n'.join(fqvocab))

    return fqvocabs


if __name__ == '__main__':
    argh.dispatch_commands([serialize2npy, filter_for_freqranges, divide_vocab_by_freqranges])
