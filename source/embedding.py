import os
from itertools import tee
import logging
import json
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


class LossLogger(CallbackAny2Vec):
    """
Callback to print loss after each epoch."""

    def __init__(self, show=False):
        """
        :param show: If True, show loss curve in the end.
        """
        self.epoch = 0
        self.prev_cum_loss = 0
        self.batch_losses = []  # loss after each batch, for every epoch (?)
        self.epoch_losses = []
        self.show = show

    def on_epoch_begin(self, model):
        self.batch = 0

    def on_epoch_end(self, model):
        cum_loss = model.get_latest_training_loss()
        eloss = cum_loss - self.prev_cum_loss
        print("Epoch #{} end, loss: {}".format(self.epoch, eloss))
        self.epoch += 1
        self.epoch_losses.append(eloss)

    def on_batch_end(self, model):
        cum_loss = model.get_latest_training_loss()
        loss = abs(cum_loss - self.prev_cum_loss)
        print(f"Epoch {self.epoch} - Batch {self.batch} end loss: {loss}")
        self.prev_cum_loss = cum_loss
        self.batch_losses.append(loss)
        self.batch += 1

    def on_train_end(self, model):
        plt.plot(self.batch_losses)
        if self.show:
            plt.show()


def train(corpus, save_path, load_path=None,
          size=300, window=5, min_count=10, workers=4,
          epochs=5, max_vocab_size=None, show_loss=False, save_loss=False):
    """
    Train w2v.
    :param corpus: list of list strings
    :param save_path: Model file path
    :return: trained model
    """
    texts, texts_build, texts_l = tee(corpus, 3)
    loss_logger = LossLogger(show_loss)  # TODO: loss curve looks weird with multiple workers

    if not os.path.exists(save_path) and load_path is None:
        model = Word2Vec(texts_build, size=size, window=window, min_count=min_count, workers=workers,
                         max_vocab_size=max_vocab_size, compute_loss=False, hs=0, sg=1, iter=epochs)
    else:
        if load_path is None:
            load_path = save_path
        print(f'Loading model {load_path}')
        model = Word2Vec.load(load_path)
        model.build_vocab(texts_build, update=True)
        logger.debug('Updates vocab, new size: {}'.format(len(model.wv.vocab)))

    model.train(texts, total_examples=model.corpus_count, epochs=model.iter, callbacks=[loss_logger], compute_loss=True)

    print('Saving model')
    model.save(save_path)
    if save_loss:
        plt.savefig(save_path + '_losscurve.png')
        with open(save_path + '_losscurve.json', 'w') as f:
            json.dump(loss_logger.batch_losses, f)

    return model
