"""
Prepare COCO datasets
==============================

`COCO <http://cocodataset.org/#home>`_ is a large-scale object detection, segmentation, and captioning datasetself.
This tutorial will walk through the steps of preparing this dataset for GluonCV.

.. image:: http://cocodataset.org/images/coco-logo.png

.. hint::

   You need 42.7 GB disk space to download and extract this dataset. SSD is
   preferred over HDD because of its better performance.

   The total time to prepare the dataset depends on your Internet speed and disk
   performance. For example, it often takes 20 min on AWS EC2 with EBS.

Prepare the dataset
-------------------

We need the following four files from `COCO <http://cocodataset.org/#download>`_:

+------------------------------------------------------------------------------------------------------------------------+--------+------------------------------------------+
| Filename                                                                                                               | Size   | SHA-1                                    |
+========================================================================================================================+========+==========================================+
| `train2017.zip <http://images.cocodataset.org/zips/train2017.zip>`_                                                    | 18 GB  | 10ad623668ab00c62c096f0ed636d6aff41faca5 |
+------------------------------------------------------------------------------------------------------------------------+--------+------------------------------------------+
| `val2017.zip <http://images.cocodataset.org/zips/val2017.zip>`_                                                        | 778 MB | 4950dc9d00dbe1c933ee0170f5797584351d2a41 |
+------------------------------------------------------------------------------------------------------------------------+--------+------------------------------------------+
| `annotations_trainval2017.zip  <http://images.cocodataset.org/annotations/annotations_trainval2017.zip>`_              | 241 MB | 8551ee4bb5860311e79dace7e79cb91e432e78b3 |
+------------------------------------------------------------------------------------------------------------------------+--------+------------------------------------------+
| `stuff_annotations_trainval2017.zip <http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip>`_   | 401 MB | e7aa0f7515c07e23873a9f71d9095b06bcea3e12 |
+------------------------------------------------------------------------------------------------------------------------+--------+------------------------------------------+

The easiest way to download and unpack these files is to download helper script
:download:`mscoco.py<../../../scripts/datasets/mscoco.py>` and run
the following command:

.. code-block:: bash
    pip install cython
    pip install pycocotools
    python mscoco.py

which will automatically download and extract the data into ``~/.mxnet/datasets/coco``.

If you already have the above files sitting on your disk,
you can set ``--download-dir`` to point to them.
For example, assuming the files are saved in ``~/coco/``, you can run:

.. code-block:: bash

   python mscoco.py --download-dir ~/coco

"""

################################################################
# Read with GluonCV
# -----------------
#
# Loading images and labels is straight-forward with
# :py:class:`gluoncv.data.COCODetection`.


from gluoncv import data
import torchvision.datasets as dset
from source import eval
import torchvision.transforms as transforms

DATADIR = '/Users/anitavero/projects/data'
MSCOCO = DATADIR + '/mscoco_mini'
val_dataset = None


def load_data():
    global val_dataset
    # train_dataset = data.COCODetection(splits=['instances_train2017'], root=MSCOCO)
    val_dataset = data.COCODetection(splits=['instances_val2017'], root=MSCOCO)
    # print('Num of training images:', len(train_dataset))
    print('Num of validation images:', len(val_dataset))


def main():
    det = dset.CocoDetection(root=MSCOCO + '/val2017',
                             annFile=MSCOCO + '/annotations/stuff_val2017.json',
                             transform=transforms.ToTensor())

    img, target = det[0]

    print("Image Size: ", img.size())
    print('\n'.join(target))

    # Get image ids for categories
    cats = det.coco.loadCats([s['category_id'] for s in target])

    print(cats)
    print(det.coco.getImgIds(catIds=cats[0]['id']))


if __name__ == "__main__":
    load_data()

    eval.load_datasets(DATADIR)
    eval.coverage(val_dataset.classes)

    ################################################################
    # Now let's visualize one example.

    # val_image, val_label = val_dataset[0]
    # bounding_boxes = val_label[:, :4]
    # class_ids = val_label[:, 4:5]
    # classes = [val_dataset.classes[int(cid)] for cid in class_ids.reshape(20)]
    # height, width, RGB = val_image.shape
    # print('Image size (height, width, RGB):', val_image.shape)
    # print('Num of objects:', bounding_boxes.shape[0])
    # print('Bounding boxes (num_boxes, x_min, y_min, x_max, y_max):\n',
    #       bounding_boxes)
    # print('Class IDs (num_boxes, ):\n', classes)
    #
    # val_dataset.coco.getImgIds(catIds=[62])
    #
    # for i in range(bounding_boxes.shape[0]):
    #     x, y, w, h = bounding_boxes[i]
    #     bbox_img = visutils.crop_bbox(val_image.asnumpy(), x, y, w, h)
    #
    #     utils.viz.plot_image(np.array(bbox_img))
    #     plt.title(classes[i])
    #     plt.show()
    #
    # utils.viz.plot_bbox(val_image.asnumpy(), bounding_boxes, scores=None,
    #                     labels=class_ids, class_names=val_dataset.classes)
    # plt.show()


##################################################################
# Finally, to use both ``train_dataset`` and ``val_dataset`` for training, we
# can pass them through data transformations and load with
# :py:class:`mxnet.gluon.data.DataLoader`, see :download:`train_ssd.py
# <../../../scripts/detection/ssd/train_ssd.py>` for more information.
