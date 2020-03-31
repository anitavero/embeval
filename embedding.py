import os
from gensim.models import Word2Vec
import argh
from itertools import tee
import logging

logger = logging.getLogger(__name__)


def train(corpus, save_path,
          size=300, window=5, min_count=10, workers=4,
          epochs=5, max_vocab_size=None):
    """
    Train w2v.
    :param corpus: list of list strings
    :param save_path: Model file path
    :return: trained model
    """
    texts, texts_build, texts_l = tee(corpus, 3)
    total_examples = len(list(texts_l))

    if not os.path.exists(save_path):
        model = Word2Vec(texts_build, size=size, window=window, min_count=min_count, workers=workers,
                         max_vocab_size=max_vocab_size, compute_loss=True)
    else:
        model = Word2Vec.load(save_path)
        model.build_vocab(texts_build, update=True)
        logger.debug('Updates vocab, new size: {}'.format(len(model.wv.vocab)))

    model.train(texts, total_examples=total_examples, epochs=epochs)

    model.save(save_path)

    return model