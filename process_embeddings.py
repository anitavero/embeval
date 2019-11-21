import os
import pickle
import numpy as np
from tqdm import tqdm
import argh


def agg_img_embeddings(filepath, savedir, maxnum=10):
    """Convert a pickled dictionary to numpy embedding end vocabulary files for eval.
        The embedding is a numpy array of shape(vocab size, vector dim)
        Vocabulary is a text file including word separated by new line.
    """

    with open(filepath, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')    # Load python2 pickles

    filename = os.path.basename(filepath).split('.')[0]

    with open(os.path.join(savedir, filename + '.vocab'), 'w') as f:
        f.write('\n'.join([str(s) for s in data_dict.keys()]))


    # Aggregate image vectors for a word, using the fist min(maxnum, imangenum) images
    embeddings = np.empty((len(data_dict), np.array(list(list(data_dict.values())[0].values())).shape[1]))
    for i, imgs in enumerate(tqdm(data_dict.values())):
        vecs = np.array(list(imgs.values()))
        aggvec = []
        for i in range(min(maxnum, vecs.shape[0])):
            embeddings[i] = vecs.mean(axis=0)

    np.save(os.path.join(savedir, filename + '.npy'), embeddings)


if __name__ == '__main__':
    argh.dispatch_command(agg_img_embeddings)
