import os, shutil
import numpy as np
import scipy.io as sio
from functools import partial

import keras.backend as K

from utils.datagen import DirectoryDataGenerator
from utils.imgprocessing import (resize_and_crop, center_crop, random_crop,
                                 meanstd, horizontal_flip, color_jitter,
                                 ten_crop)


mean = np.asarray([119.26753706, 115.92306357, 116.10504895], dtype=K.floatx())
std = np.asarray([75.48790007, 75.23135039, 77.03315339], dtype=K.floatx())

LOAD_DIM = 256
TRAIN_DIM = CROP_DIM = 224
NUM_CLASSES = 196
URL = './data/images/cars/'


def distribute_images(folder, annotation_file):
    """
    Split a folder of images to a subdirectory per class 
    based on an annotation file.
    """
    # get annotations
    annotations = sio.loadmat(annotation_file)['annotations'][0]

    for ann in annotations:

        label = ann[4].flatten()[0]
        fname = ann[5].flatten()[0]

        imgpath = os.path.join(folder, fname)
        subfolder = os.path.join(folder, "{0:0>3}".format(label))

        # check if the subdirectory for this folder already exists
        if not os.path.isdir(subfolder):
            os.makedirs(subfolder)

        # move image to its subfolder
        shutil.move(imgpath, subfolder)


def get_data_gen():
    """
    Return train and val generators that give data in batches,
    and data label names.
    """
    # define preprocessing pipeline
    train_transforms = [
        partial(resize_and_crop, new_size=LOAD_DIM),
        partial(color_jitter, brightness=0.4, contrast=0.4, saturation=0.4),
        partial(meanstd, mean=mean, std=std),
        partial(random_crop, new_size=CROP_DIM, padding=10),
        partial(horizontal_flip, f=0.5),
    ]

    # data generators
    train_generator = DirectoryDataGenerator(
        os.path.join(URL, 'train'), train_transforms, shuffle=True)

    val_generator = get_test_gen('val')

    return train_generator, val_generator


def get_test_gen(datatype='val'):

    crop = ten_crop if datatype == 'test' else center_crop

    transforms = [
        partial(resize_and_crop, new_size=LOAD_DIM),
        partial(meanstd, mean=mean, std=std),
        partial(crop, new_size=CROP_DIM)
    ]

    generator = DirectoryDataGenerator(
        os.path.join(URL, datatype), transforms, shuffle=False)

    return generator
