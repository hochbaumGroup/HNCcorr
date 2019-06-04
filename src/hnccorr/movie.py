import numpy as np
from PIL import Image
from PIL.TiffTags import TAGS

from hnccorr.utils import list_images


class Movie:
    """2-dimensional calcium imaging movie stored in memory.

    Attributes:
        _data (np.array): Fluorescence data. Array has size T x N1 x N2. T is
            the number of frame (num_frames), N1 and N2 are the number of
            pixels in the first and second dimension respectively.
        _data_size (tuple): Size of array _data.
        name(str): Name of experiment
        num_frames (int): Number of frames in movie
    """

    def __init__(self, name, data):
        self.name = name
        self._data = data
        self.data_size = data.shape

    @classmethod
    def from_tiff_images(cls, name, image_dir, num_images):
        data = cls._load_images(image_dir, num_images)
        return cls(name, data)

    @staticmethod
    def _load_images(image_dir, num_images):
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
        with Image.open(first_image) as image:
            meta = {TAGS[key]: image.tag[key] for key in image.tag}

        # set size of data
        data_size = (len(images), meta["ImageLength"][0], meta["ImageWidth"][0])

        data = np.zeros(data_size, np.uint16)

        for i, filename in enumerate(images):
            with Image.open(filename) as image:
                data[i, :, :] = np.array(image)
        return data

    def __getitem__(self, key):
        """Access data directly from underlying numpy array"""
        return self._data.__getitem__(key)

    def is_valid_pixel_index(self, index):
        if self.num_dimensions == len(index):
            zero_tuple = (0,) * self.num_dimensions
            for i, lower, upper in zip(index, zero_tuple, self.pixel_size):
                if i < lower or i >= upper:
                    return False
            return True
        return False

    @property
    def num_frames(self):
        return self.data_size[0]

    @property
    def pixel_size(self):
        return self.data_size[1:]

    @property
    def num_pixels(self):
        return np.product(self.data_size[1:])

    @property
    def num_dimensions(self):
        return len(self.data_size[1:])
