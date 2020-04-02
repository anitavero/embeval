import os
from itertools import tee
import logging
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

logger = logging.getLogger(__name__)


class EpochLogger(CallbackAny2Vec):
    """Callback to print loss after each epoch."""

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        loss = model.get_latest_training_loss()
        print("Epoch #{} start, loss: {}".format(self.epoch, loss))

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        print("Epoch #{} end, loss: {}".format(self.epoch, loss))
        self.epoch += 1


def train(corpus, save_path, load_path=None,
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

    epoch_logger = EpochLogger()

    if not os.path.exists(save_path) and load_path is None:
        model = Word2Vec(texts_build, size=size, window=window, min_count=min_count, workers=workers,
                         max_vocab_size=max_vocab_size, compute_loss=False, hs=0, sg=1)
    else:
        if load_path is None:
            load_path = save_path
        print(f'Loading model {load_path}')
        model = Word2Vec.load(load_path)
        model.build_vocab(texts_build, update=True)
        logger.debug('Updates vocab, new size: {}'.format(len(model.wv.vocab)))

    model.train(texts, total_examples=total_examples, epochs=epochs, callbacks=[epoch_logger], compute_loss=True)
    print('Loss after training:', model.get_latest_training_loss())

    print('Saving model')
    model.save(save_path)

    return model