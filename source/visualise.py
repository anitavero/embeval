import os
import argh
from argh import arg
import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector

from source.process_embeddings import Embeddings, filter_by_vocab
from source.unsupervised_metrics import wn_category


@arg('--tn-label', choices=['frequency',
                            'optics_cl'])
def tensorboard_emb(data_dir, model_name, output_path, tn_label='clusters', label_name='clusters'):
    """
    Visualise embeddings using TensorBoard.
    Code from: https://gist.github.com/BrikerMan/7bd4e4bd0a00ac9076986148afc06507
    :param model_name: name of numpy array files: embedding (.npy) and vocab (.vocab)
    :param output_path: str, directory
    :param tn_label: function(word) returns value, labels for text and/or colouring
    :param label_name: str, title for the labeling (e.g.: Cluster)

    Usage on remote server with port forwarding:
        * when you ssh into the machine, you use the option -L to transfer the port 6006 of the remote server
          into the port 16006 of my machine (for instance): 
        * ssh -L 16006:127.0.0.1:6006 alv34@yellowhammer
          What it does is that everything on the port 6006 of the server (in 127.0.0.1:6006) will be forwarded 
          to my machine on the port 16006.
        * You can then launch tensorboard on the remote machine using a standard tensorboard --logdir log with
          the default 6006 port
        * On your local machine, go to http://127.0.0.1:16006 and enjoy your remote TensorBoard.
    """
    if tn_label == 'None':
        labeler = None
    elif tn_label == 'clusters':
        labeler = lambda w: wn_category(w)

    print('Load embedding')
    embs = Embeddings(data_dir, [model_name])
    if labeler:
        print('Filter embedding and vocab by existing cluster names')
        filter_vocab = [w for w in embs.vocabs[0] if labeler(w) is not None]
        model, vocab = filter_by_vocab(embs.embeddings[0], embs.vocabs[0], filter_vocab)
        print('#Vocab after filtering:', len(vocab))
    else:
        model = embs.embeddings[0]
        vocab = embs.vocabs[0]

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
    checkpoint.save(os.path.join(output_path, f"embedding.ckpt"))

    # Set up config
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
    embedding.tensor_name = f"embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = meta_file
    projector.visualize_embeddings(output_path, config)

    print('Run `tensorboard --logdir={0}` to run visualize result on tensorboard'.format(output_path))


if __name__ == '__main__':
    argh.dispatch_command(tensorboard_emb)
