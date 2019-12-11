import os
import pickle
import json
import numpy as np
from tqdm import tqdm
import argh
from typing import List
from itertools import combinations
from utils import get_vec


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


if __name__ == '__main__':
    argh.dispatch_command(serialize2npy)
