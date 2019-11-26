import numpy as np


def join_struct_arrays(arrays):
    new_dt = list(set(sum([a.dtype.descr for a in arrays], [])))
    joint = np.zeros(arrays[0].shape, dtype=new_dt)
    for a in arrays:
        for nm in a.dtype.names:
            joint[nm] = a[nm]
    return joint


def get_vec(word, embeddings, vocab):
    return embeddings[np.where(vocab == word)[0][0]].reshape(1, -1)