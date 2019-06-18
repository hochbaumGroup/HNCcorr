import numpy as np
from PIL import Image
from PIL.TiffTags import TAGS

from hnccorr.utils import list_images


class Movie:
    """Calcium imaging movie class.

    Data is stored in an in-memory numpy array. Class supports both 2- and 3-
    dimensional movies.

    Attributes:
        name(str): Name of the experiment.
        _data (np.array): Fluorescence data. Array has size T x N1 x N2. T is
            the number of frame (num_frames), N1 and N2 are the number of
            pixels in the first and second dimension respectively.
        _data_size (tuple): Size of array _data.
    """

    def __init__(self, name, data):
        self.name = name
        self._data = data
        self.data_size = data.shape

    @classmethod
    def from_tiff_images(cls, name, image_dir, num_images):
        """Loads tiff images into numpy array.

        Data is assumed to be stored in 16-bit unsigned integers. Frame numbers are assumed to be padded with zeros: 00000, 00001, 00002, etc. This is required such that Python sorts the images correctly. Frame numbers can start from 0, 1, or any other number. Files must have the extension ``.tiff``.

        Args:
            name (str): Movie name.
            image_dir (str): Path of image folder.
            num_images (int): Number of images in the folder.

        Returns:
            Movie: Movie created from image files.
        """
        data = cls._load_tiff_images(image_dir, num_images)
        return cls(name, data)

    @staticmethod
    def _load_tiff_images(image_dir, num_images):
        """Loads tiff images into numpy array.

        Data is assumed to be stored in 16-bit unsigned integers. Frame numbers are assumed to be padded with zeros: 00000, 00001, 00002, etc. This is required such that Python sorts the images correctly. Frame numbers can start from 0, 1, or any other number. Files must have the extension ``.tiff``.

        Args:
            image_dir (str): Path of image folder.
            num_images (int): Number of images in the folder.

        Returns:
            np.array: Data with shape (T, N_1, N_2, N_3) where T is # of images.
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
        """Provides direct access to the movie data.

        Movie is stored in array with shape (T, N_1, N_2, ...), where T is the number
        of frames in the movie. N_1, N_2, ... are the number of pixels in the first
        dimension, second dimension, etc.

        Args:
            key (tuple): Valid index for a numpy array.

        Returns:
            np.array
        """
        return self._data.__getitem__(key)

    def is_valid_pixel_coordinate(self, coordinate):
        """Checks if coordinate is a coordinate for a pixel in the movie."""
        if self.num_dimensions != len(coordinate):
            return False

        zero_tuple = (0,) * self.num_dimensions
        for i, lower, upper in zip(coordinate, zero_tuple, self.pixel_shape):
            if not lower <= i < upper:
                return False
        return True

    @property
    def num_frames(self):
        """Number of frames in the movie."""
        return self.data_size[0]

    @property
    def pixel_shape(self):
        """Resolution of the movie in pixels."""
        return self.data_size[1:]

    @property
    def num_pixels(self):
        """Number of pixels in the movie."""
        return np.product(self.data_size[1:])

    @property
    def num_dimensions(self):
        """Dimension of the movie (excludes time dimension)."""
        return len(self.data_size[1:])

    def extract_valid_pixels(self, pixels):
        """Returns subset of pixels that are valid coordinates for the movie."""
        return {pixel for pixel in pixels if self.is_valid_pixel_coordinate(pixel)}
