import os
import argh
from argh import arg
import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector

from source.process_embeddings import Embeddings
from source.unsupervised_metrics import wn_category


@arg('--tn-label', choices=['frequency',
                            'optics_cl'])
def tensorboard_emb(data_dir, model_name, output_path, tn_label='optics_cl', label_name='optics_cl'):
    """
    Visualise embeddings using TensorBoard.
    Code from: https://gist.github.com/BrikerMan/7bd4e4bd0a00ac9076986148afc06507
    :param model_name: name of numpy array files: embedding (.npy) and vocab (.vocab)
    :param model_name: str, name for meta files
    :param output_path: str, directory
    :param tn_label: function(word) returns value, labels for text and/or colouring
    :param label_name: str, title for the labeling (e.g.: Cluster)
    """
    embs = Embeddings(data_dir, [model_name])
    model, vocab = embs.embeddings[0], embs.vocabs[0]
    if tn_label == 'frequency':
        pass
    elif tn_label == 'optics_cl':
        labeler = lambda w: wn_category(w)

    file_name = "{}_metadata".format(model_name)
    meta_file = "{}.tsv".format(file_name)
    placeholder = np.zeros((len(vocab), model.shape[1]))

    with open(os.path.join(output_path, meta_file), 'wb') as file_metadata:
        file_metadata.write("Word\t{}".format(label_name).encode('utf-8') + b'\n')
        for i, word in enumerate(vocab):
            placeholder[i] = model[i, :]
            # temporary solution for https://github.com/tensorflow/tensorflow/issues/9094
            if word == '':
                print("Emply Line, should replecaed by any thing else, or will cause a bug of tensorboard")
                file_metadata.write("{0}".format('<Empty Line>').encode('utf-8') + b'\n')
            else:
                file_metadata.write(
                    "{0}\t{1}".format(word, labeler(word)).encode('utf-8') + b'\n')

    weights = tf.Variable(placeholder, trainable=False, name=file_name)
    checkpoint = tf.train.Checkpoint(embedding=weights)
    checkpoint.save(os.path.join(output_path, "embedding.ckpt"))

    # Set up config
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = meta_file
    projector.visualize_embeddings(output_path, config)

    print('Run `tensorboard --logdir={0}` to run visualize result on tensorboard'.format(output_path))


if __name__ == '__main__':
    argh.dispatch_command(tensorboard_emb)
