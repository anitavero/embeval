import sys, os
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/source')

from source.unsupervised_metrics import *
import numpy as np


def test_order_words_by_centroid_distance():
    clusters = [(0, ['c', 'b', 'a']), (1, ['e', 'd'])]
    cl_labels = {'a': 0, 'b': 0.1, 'c': 1, 'd': 0.1, 'e': 0.2}
    cluster_label_filepath = 'test/data/test_cluster_labels.json'
    with open('test/data/dists_from_centr_labels.json', 'w') as f:
        json.dump(cl_labels, f)

    order_words_by_centroid_distance(clusters, cluster_label_filepath)
    assert clusters == [(0, ['a', 'b', 'c']), (1, ['d', 'e'])]


def test_distances_from_centroids():
    emb = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 1, 1]])
    vocab = ['a', 'b', 'c', 'd']
    centroids = np.array([[2, 0, 0],
                          [0, 4, 0]])
    label_dict = {'a': 0, 'b': 0, 'c': 1, 'd': 1}

    dists = distances_from_centroids(emb, vocab, label_dict, centroids)
    assert dists['a'] == 0.0
    assert dists['b'] == 1.0
    assert dists['c'] == 1.0
    assert np.isclose(dists['d'], 1 - (4 / (np.sqrt(3) * np.sqrt(16))))


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