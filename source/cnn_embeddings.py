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

from process_embeddings import serialize2npy


@arg('-cnn', '--cnn_model', choices=['alexnet', 'resnet-18', 'googlenet'], default='resnet-18')   # TODO
def get_cnn(image_dir, word_index_file, cnn_model='resnet', agg_maxnum=10):
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
    repr_path = os.path.join(image_dir, cnn_model + '.pkl')
    with open(repr_path, 'wb') as f:
        pickle.dump(word_img_repr, f)

    # Save  aggregated embeddings
    serialize2npy(repr_path, image_dir, agg_maxnum)


if __name__ == '__main__':
    argh.dispatch_command(get_cnn)
