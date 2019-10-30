import os, re
import numpy as np
from PIL import Image
import mxnet

def crop_bbox(image, x, y, w, h):
    """
    Crops out a bounding box from image.
    :param image: PIL Image
    :param x:
    :param y:
    :param w:
    :param h:
    :return: PIL Image
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(np.uint8(image))
    return image.crop((x, y, w, h))


def save_crop(image, x, y, w, h, fname, savedir, skip_existing=True):
    img = crop_bbox(image, x, y, w, h)
    image_file = os.path.join(savedir, fname)
    if os.path.exists(image_file):
        if not skip_existing:
            image_file = re.sub('.jpg', 'I.jpg', image_file)
            img.save(image_file)
    else:
        img.save(image_file)