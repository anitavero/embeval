from source.process_embeddings import *


def test_midfusion():

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