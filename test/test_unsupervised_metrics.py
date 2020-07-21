import sys, os
sys.path.append(os.getcwd() + '/source')

from source.unsupervised_metrics import *


def test_n_nearest_neighbors():
    vocab = np.array(['a', 'b', 'c', 'd', 'e'])
    words = np.array(['a', 'c', 'e'])
    E = np.array([[1, 0],
                  [0, 1],
                  [1, 1],
                  [1, 0.5],
                  [0.5, 1]])


    NN = n_nearest_neighbors(words, E, vocab, n=1)
    assert (NN[0, :] == words).all()
    assert (NN[1:, 0] == np.array(['d'])).all()
    assert (NN[1:, 1] == np.array(['d'])).all()

    NN = n_nearest_neighbors(words, E, vocab, n=3)
    assert (NN[0, :] == words).all()
    assert (NN[1:, 0] == np.array(['d', 'c', 'e'])).all()
    assert (NN[1:, 1] == np.array(['d', 'e', 'a'])).all()

    words = np.array(['b'])
    NN = n_nearest_neighbors(words, E, vocab, n=3)
    assert (NN[0, :] == words).all()
    assert (NN[1:, 0] == np.array(['e', 'c', 'd'])).all()

    words = np.array([])
    NN = n_nearest_neighbors(words, E, vocab, n=3)
    assert NN.size == 0