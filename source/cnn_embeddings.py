import sys, os
sys.path.append("../../img2vec/img2vec_pytorch")  # Adds higher directory to python modules path.
from img_to_vec import Img2Vec
from PIL import Image
import json
import pickle
from tqdm import tqdm
import argh
from argh import arg
from collections import defaultdict
from glob import glob

from process_embeddings import serialize2npy

# TODO: more models
@arg('-cnn', '--cnn_model', choices=['alexnet', 'resnet-18', 'googlenet'], default='resnet-18')
def get_cnn(image_dir, word_index_file, savedir=None, cnn_model='resnet', agg_maxnum=10):
    """Extract CNN representations for images in a directory and saves it into a dictionary file."""
    img2vec = Img2Vec(model=cnn_model)

    # Dictionary of {words: {img_name: img_representation}}
    word_img_repr = defaultdict(dict)

    with open(word_index_file) as f:
        word_imgs = json.load(f)

    for word, img_names in tqdm(word_imgs.items()):
        for imgn in img_names:
            img = Image.open(os.path.join(image_dir, imgn))
            word_img_repr[word][imgn] = img2vec.get_vec(img)

    # Save representations
    if savedir is None:
        savedir = image_dir

    repr_path = os.path.join(savedir, cnn_model + '.pkl')
    with open(repr_path, 'wb') as f:
        pickle.dump(word_img_repr, f)

    # Save aggregated embeddings
    serialize2npy(repr_path, savedir, agg_maxnum)


def create_index_from_fnames(image_dir, savepath):
    files = glob(image_dir)
    word_index_file = defaultdict(list)
    for f in files:
        fname = os.path.basename(f)
        word = fname.split('_')[0]
        word_index_file[word].append(fname)
    with open(savepath, 'wb') as f:
        json.dump(word_index_file, f)


if __name__ == '__main__':
    argh.dispatch_command(get_cnn)
