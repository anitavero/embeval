import sys, os
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/source')

from source.unsupervised_metrics import *


def test_get_clustering_labels_metrics():
    """
    Test data which is loaded from 'test/data':
        test_model = np.array([[1, 1, 1],
                               [2, 2, 2],
                               [3, 3, 3],
                               [4, 4, 4]])
        test_vocab = ['a', 'b', 'c', 'd']
    """
    get_clustering_labels_metrics(['test_model'], datadir='test/data/',
                                  savedir='test/data/',
                                  cluster_method='kmeans', n_clusters=4, random_state=1, eps=0.5, min_samples=90,
                                  workers=1, suffix='')
    labels = np.load('test/data/cluster_labels_kmeans_test_model_nc4.npy')
    print(labels)
    assert len(set(labels)) == 4

    get_clustering_labels_metrics(['test_model'], datadir='test/data/',
                                  savedir='test/data/',
                                  cluster_method='kmeans', n_clusters=3, random_state=1, eps=0.5, min_samples=90,
                                  workers=1, suffix='')
    labels = np.load('test/data/cluster_labels_kmeans_test_model_nc3.npy')
    print(labels)
    assert len(set(labels)) == 3

    get_clustering_labels_metrics(['test_model'], datadir='test/data/',
                                  savedir='test/data/',
                                  cluster_method='kmeans', n_clusters=2, random_state=1, eps=0.5, min_samples=90,
                                  workers=1, suffix='')
    labels = np.load('test/data/cluster_labels_kmeans_test_model_nc2.npy')
    print(labels)
    assert len(set(labels)) == 2


def test_n_nearest_neighbors():
    vocab = np.array(['a', 'b', 'c', 'd', 'e'])
    words = np.array(['a', 'c', 'e'])
    E = np.array([[1, 0],
                  [0, 1],
                  [1, 1],
                  [1, 0.5],
                  [0.5, 1]])


    NN = get_n_nearest_neighbors(words, E, vocab, n=1)
    assert (NN[0, :] == words).all()
    assert (NN[1:, 0] == np.array(['d'])).all()
    assert (NN[1:, 1] == np.array(['d'])).all()

    NN = get_n_nearest_neighbors(words, E, vocab, n=3)
    assert (NN[0, :] == words).all()
    assert (NN[1:, 0] == np.array(['d', 'c', 'e'])).all()
    assert (NN[1:, 1] == np.array(['d', 'e', 'a'])).all()

    words = np.array(['b'])
    NN = get_n_nearest_neighbors(words, E, vocab, n=3)
    assert (NN[0, :] == words).all()
    assert (NN[1:, 0] == np.array(['e', 'c', 'd'])).all()

    words = np.array([])
    NN = get_n_nearest_neighbors(words, E, vocab, n=3)
    assert NN.size == 0