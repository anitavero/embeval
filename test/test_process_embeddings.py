import sys, os
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/source')

from source.process_embeddings import *


data_dir = 'test/data/'


def test_filter_for_freqrange():
    """
    test vecs = np.array([[1, 1, 1],
                          [2, 2, 2],
                          [3, 3, 3],
                          [4, 4, 4]])
    test vocab = ['a', 'b', 'c', 'd']
    dist = {"a": 1, "b": 10, "c": 20, "d": 40}
    """
    fq_ranges = [(0, 15), (30, 100), (50, 60), (1, 40)]
    fembs = filter_for_freqrange(data_dir, ['test_model'], data_dir + '/dist.json', fq_ranges)

    assert (fembs[f'{0} {15}']['vecs'] == np.array([[1, 1, 1],
                                                    [2, 2, 2]])).all()
    assert fembs[f'{0} {15}']['vocab'] == ['a', 'b']

    assert (fembs[f'{30} {100}']['vecs'] == np.array([[4, 4, 4]])).all()
    assert fembs[f'{30} {100}']['vocab'] == ['d']

    assert (fembs[f'{50} {60}']['vecs'] == np.empty((0, 3))).all()
    assert fembs[f'{50} {60}']['vocab'] == []

    assert (fembs[f'{1} {40}']['vecs'] == np.array([[1, 1, 1],
                                                    [2, 2, 2],
                                                    [3, 3, 3],
                                                    [4, 4, 4]])).all()
    assert fembs[f'{1} {40}']['vocab'] == ['a', 'b', 'c', 'd']


def test_filter_by_vocab():
    emb = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                    [10, 11, 12]])
    vocab = np.array(['a', 'b', 'c', 'd'])

    filter_vocab = ['a']
    femb, fvocab = filter_by_vocab(emb, vocab, filter_vocab)
    assert fvocab == ['a']
    assert (femb == np.array([[1, 2, 3]])).all()

    filter_vocab = []
    femb, fvocab = filter_by_vocab(emb, vocab, filter_vocab)
    assert fvocab == []
    assert (femb == np.empty((0, 3))).all()

    filter_vocab = ['b', 'd']
    femb, fvocab = filter_by_vocab(emb, vocab, filter_vocab)
    assert fvocab == ['b', 'd']
    assert (femb == np.array([[4, 5, 6],
                             [10, 11, 12]])).all()

    filter_vocab = vocab
    femb, fvocab = filter_by_vocab(emb, vocab, filter_vocab)
    assert (fvocab == vocab).all()
    assert (femb == emb).all()


def test_mid_fusion():

    emb1 = np.array([[1, 2, 3],
                    [4, 5, 6]])
    emb2 = np.array([[7, 8, 9],
                    [10, 11, 12]])
    vocab1 = np.array(['a', 'b'])
    vocab2 = np.array(['b', 'c'])
    labels = np.array(['one', 'two'])

    mm_embs, mm_vocabs, mm_labels = mid_fusion((emb1, emb2), (vocab1, vocab2), labels, padding=True)
    assert len(mm_vocabs) == 1
    assert len(mm_embs) == 1
    for i, word in enumerate(mm_vocabs[0]):
        if np.isin(word, vocab1).any():
            id1 = np.argwhere(vocab1 == word)[0][0]
            assert (mm_embs[0][i, :3] == emb1[id1]).all()
        if np.isin(word, vocab2).any():
            id2 = np.argwhere(vocab2 == word)[0][0]
            assert (mm_embs[0][i, 3:] == emb2[id2]).all()

    ####### Test no padding

    mm_embs, mm_vocabs, mm_labels = mid_fusion((emb1, emb2), (vocab1, vocab2), labels, padding=False)
    assert len(mm_vocabs) == 1
    assert len(mm_embs) == 1
    assert (mm_vocabs[0] == np.array(['b'])).all()
    assert (mm_embs[0] == np.array([[4, 5, 6, 7, 8, 9]])).all()


    ########## Test order matching ##########

    emb1 = np.array([[1, 1],
                     [2, 2],
                     [3, 3]])
    emb2 = np.array([[11, 11],
                     [22, 22],
                     [33, 33]])
    vocab1 = np.array(['a', 'b', 'c'])
    vocab2 = np.array(['c', 'a', 'b'])
    labels = np.array(['one', 'two'])

    mm_embs, mm_vocabs, mm_labels = mid_fusion((emb1, emb2), (vocab1, vocab2), labels, padding=True)
    assert len(mm_vocabs) == 1
    assert len(mm_embs) == 1
    assert (mm_embs[0][np.argwhere(mm_vocabs[0] == 'a')[0][0]] == np.array([1, 1, 22, 22])).all()
    assert (mm_embs[0][np.argwhere(mm_vocabs[0] == 'b')[0][0]] == np.array([2, 2, 33, 33])).all()
    assert (mm_embs[0][np.argwhere(mm_vocabs[0] == 'c')[0][0]] == np.array([3, 3, 11, 11])).all()

    ####### No padding should result an equivalent embedding-vocab, however not necessarily the
    ####### "same" in terms of ordering.

    mm_embs_np, mm_vocabs_np, mm_labels_np = mid_fusion((emb1, emb2), (vocab1, vocab2), labels, padding=False)
    assert (mm_embs_np[0][np.argwhere(mm_vocabs_np[0] == 'a')[0][0]] == np.array([1, 1, 22, 22])).all()
    assert (mm_embs_np[0][np.argwhere(mm_vocabs_np[0] == 'b')[0][0]] == np.array([2, 2, 33, 33])).all()
    assert (mm_embs_np[0][np.argwhere(mm_vocabs_np[0] == 'c')[0][0]] == np.array([3, 3, 11, 11])).all()
