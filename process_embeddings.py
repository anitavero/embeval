import os
import pickle
import numpy as np
from tqdm import tqdm
import argh
from typing import List
from itertools import combinations
from utils import get_vec


def agg_img_embeddings(filepath: str, savedir: str, maxnum: int = 10):
    """Convert a pickled dictionary to numpy embedding end vocabulary files for eval.
        The embedding is a numpy array of shape(vocab size, vector dim)
        Vocabulary is a text file including word separated by new line.
    """

    with open(filepath, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')  # Load python2 pickles

    filename = os.path.basename(filepath).split('.')[0]

    with open(os.path.join(savedir, filename + '.vocab'), 'w') as f:
        f.write('\n'.join([str(s, 'utf-8') for s in data_dict.keys()]))

    # Aggregate image vectors for a word, using the fist min(maxnum, imangenum) images
    embeddings = np.empty((len(data_dict), np.array(list(list(data_dict.values())[0].values())).shape[1]))
    for i, imgs in enumerate(tqdm(data_dict.values())):
        vecs = np.array(list(imgs.values()))
        embeddings[i] = vecs[:min(maxnum, vecs.shape[0])].mean(axis=0)

    np.save(os.path.join(savedir, filename + '.npy'), embeddings)


def mid_fusion(embeddings, vocabs, labels,
               padding: bool, combnum: int = 2) -> (List[np.ndarray], List[str]):
    """Concatenate embeddings pairwise for words in the intersection of their vocabulary.
        :param embeddings: List[np.ndarray] or List[Tuple[np.ndarray]]
        :param vocabs: List[np.ndarray] or List[Tuple[np.ndarray]]
        :param labels: List[np.ndarray] or List[Tuple[np.ndarray]]
        :param padding: If true all the vectors are kept from the embedding with the biggest
                        vocabulary. The vectors without a vector from another modality are
                        padded with zeros.
        :param combnum: number of modalities concatenated in the final multi-modal vector
    """
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
    for emb1, emb2, vocab1, vocab2, label1, label2 in zip(emb_pairs, vocab_pairs, label_pairs):
        shape1 = emb1.shape[1]
        shape2 = emb2.shape[1]
        label = '-'.join([label1, label2])
        if padding:
            mm_vocab = list(set(vocab1).union(set(vocab2)))
            mm_embedding = np.zeros((len(mm_vocab), shape1 + shape2))
            for w in mm_vocab:
                try:
                    mm_embedding[mm_vocab.index(w), :shape1] = get_vec(w, emb1, vocab1)
                except IndexError:
                    pass    # If the embedding doesn't have this word leave the first vector part full zeros
                try:
                    mm_embedding[mm_vocab.index(w), shape1:] = get_vec(w, emb2, vocab2)
                except IndexError:
                    pass    # If the embedding doesn't have this word leave the second vector part full zeros
        else:
            mm_vocab = list(set(vocab1).intersection(set(vocab2)))
            mm_embedding = np.zeros((len(mm_vocab), shape1 + shape2))
            for w in mm_vocab:
                mm_embedding[mm_vocab.index(w), :shape1] = get_vec(w, emb1, vocab1)
                mm_embedding[mm_vocab.index(w), shape1:] = get_vec(w, emb2, vocab2)

        mm_embeddings.append(mm_embedding)
        mm_vocabs.append(mm_vocab)
        mm_labels.append(label)

        assert mm_embedding.shape == (emb1.shape[0] + emb2.shape[0], len(mm_vocab))

    return mm_embeddings, mm_vocabs, mm_labels


if __name__ == '__main__':
    argh.dispatch_command(agg_img_embeddings)
