import sys, os
sys.path.append("../../img2vec_privatefork/img2vec_pytorch")  # Adds higher directory to python modules path.
sys.path.append("../img2vec_privatefork/img2vec_pytorch")  # Adds higher directory to python modules path.
from img_to_vec import Img2Vec
from PIL import Image
import json
import pickle
from tqdm import tqdm
import argh
from argh import arg
from collections import defaultdict
from glob import glob
import warnings

from process_embeddings import serialize2npy


cnn_models = ['alexnet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

@arg('-cnn', '--cnn_model', choices=cnn_models, default='resnet18')
def get_cnn(image_dir, word_index_file, savedir=None, cnn_model='resnet18', agg_maxnum=10, gpu=False,
            filename_prefix=''):
    """Extract CNN representations for images in a directory and saves it into a dictionary file."""
    img2vec = Img2Vec(model=cnn_model, cuda=gpu)

    # Dictionary of {words: {img_name: img_representation}}
    word_img_repr = defaultdict(dict)

    with open(word_index_file) as f:
        word_imgs = json.load(f)

    for word, img_names in tqdm(word_imgs.items()):
        for imgn in img_names:
            try:
                img = Image.open(os.path.join(image_dir, imgn)).convert('RGB')
                word_img_repr[word][imgn] = img2vec.get_vec(img)
            except FileNotFoundError:
                warnings.warn(f'Image {imgn} for word "{word}" is missing.')

    # Save representations
    if savedir is None:
        savedir = image_dir

    repr_path = os.path.join(savedir, '_'.join([filename_prefix, cnn_model + '.pkl']))
    with open(repr_path, 'wb') as f:
        pickle.dump(word_img_repr, f)

    # Save aggregated embeddings
    serialize2npy(repr_path, savedir, agg_maxnum)


def create_index_from_fnames(image_dir, savepath):
    files = glob(image_dir + '/*.jpg')
    word_index_file = defaultdict(list)
    for f in files:
        fname = os.path.basename(f)
        word = fname.split('_')[0]
        word_index_file[word].append(fname)
    with open(savepath, 'w') as f:
        json.dump(word_index_file, f)


if __name__ == '__main__':
    argh.dispatch_command(get_cnn)
