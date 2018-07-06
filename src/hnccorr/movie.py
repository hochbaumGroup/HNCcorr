import glob
import os
import numpy as np
from PIL import Image
from PIL.TiffTags import TAGS

from hnccorr.utils import list_images


class Movie(object):
    """2-dimensional calcium imaging movie stored in memory.

    Attributes:
        _data (np.array): Fluorescence data. Array has size T x N1 x N2. T is
            the number of frame (num_frames), N1 and N2 are the number of
            pixels in the first and second dimension respectively.
        _data_size (tuple): Size of array _data.
        name(str): Name of experiment
        num_frames (int): Number of frames in movie
    """

    def __init__(self, name, image_dir, num_images):
        self.name = name
        self.num_frames = num_images

        self._load_images(image_dir, num_images)

    def _load_images(self, image_dir, num_images):
        """Load images from directory.

        Sorts tiff files in the directory `image_dir` and loads the images into
        a numpy array.

        Args:
            image_dir (str): Path of image folder
            num_images (int): Number of images in the folder.
        """
        images = list_images(image_dir)

        assert len(images) == num_images

        # read image meta data
        first_image = images[0]
        with Image.open(first_image) as im:
            meta = {TAGS[key]: im.tag[key] for key in im.tag}

        if meta["BitsPerSample"][0] != 16:
            raise ValueError("Only 16 bit images are currently supported")

        # set size of data
        self.data_size = (
            len(images),
            meta["ImageLength"][0],
            meta["ImageWidth"][0],
        )

        self._data = np.zeros(self.data_size, np.uint16)

        for i, filename in enumerate(images):
            with Image.open(filename) as im:
                self._data[i, :, :] = np.array(im)

    def __getitem__(self, key):
        """Access data directly from underlying numpy array"""
        return self._data.__getitem__(key)

    def is_valid_pixel_index(self, index):
        if self.num_dimensions == len(index):
            zero_tuple = (0,) * self.num_dimensions
            for i, l, u in zip(index, zero_tuple, self.pixel_size):
                if i < l or i >= u:
                    return False
            return True
        else:
            return False

    @property
    def pixel_size(self):
        return self.data_size[1:]

    @property
    def num_pixels(self):
        return np.product(self.data_size[1:])

    @property
    def num_dimensions(self):
        return len(self.data_size[1:])
