"""
This module contains code adapted from the open-source yolov5 project, simplified for our specific use case
"""
import os
import glob
from pathlib import Path
import logging

import numpy as np
import cv2

from yolov5.utils.augmentations import letterbox
from yolov5.utils.datasets import IMG_FORMATS
from .constants import DEFAULT_IMAGE_DIM


LOGGER = logging.getLogger(__name__)


class LoadImagesMulti:  # for inference
    def __init__(self, paths, img_size=DEFAULT_IMAGE_DIM[0], stride=32, auto=True):
        # fetch list of images from provided paths

        if not isinstance(paths, list):
            paths = [paths]
        images = []
        for p in paths:
            p = str(Path(p).resolve())  # os-agnostic absolute path
            if '*' in p:
                files = sorted(glob.glob(p, recursive=True))  # glob
            elif os.path.isdir(p):
                files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
            elif os.path.isfile(p):
                files = [p]  # files
            else:
                raise Exception(f'ERROR: {p} does not exist')
            images.extend([x for x in files if x.split('.')[-1].lower() in IMG_FORMATS])
        num_images = len(images)

        self.img_size = img_size
        self.stride = stride
        self.image_files = images
        self.num_images = num_images  # number of files
        self.mode = 'image'
        self.auto = auto
        assert self.num_images > 0, f'No images found in {paths}. Supported formats are:\nimages: {IMG_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.num_images:
            raise StopIteration
        path = self.image_files[self.count]

        # Read image
        self.count += 1
        img0 = cv2.imread(path)  # BGR
        assert img0 is not None, 'Image Not Found ' + path

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return path, img, img0

    def __len__(self):
        return self.num_images
