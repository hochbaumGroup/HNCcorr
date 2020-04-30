# Copyright © 2017. Regents of the University of California (Regents). All Rights
# Reserved.
#
# Permission to use, copy, modify, and distribute this software and its documentation
# for educational, research, and not-for-profit purposes, without fee and without a
# signed licensing agreement, is hereby granted, provided that the above copyright
# notice, this paragraph and the following two paragraphs appear in all copies,
# modifications, and distributions. Contact The Office of Technology Licensing, UC
# Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201,
# for commercial licensing opportunities. Created by Quico Spaen, Roberto Asín-Achá,
# and Dorit S. Hochbaum, Department of Industrial Engineering and Operations Research,
# University of California, Berkeley.
#
# IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
# INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE
# OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE
# SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS
# IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES,
# ENHANCEMENTS, OR MODIFICATIONS.
"""Components for calcium-imaging movies in HNCcorr."""

import os
import math
import numpy as np
from PIL import Image
from PIL.TiffTags import TAGS

from hnccorr.utils import (
    add_offset_to_coordinate,
    add_offset_set_coordinates,
    add_time_index,
    generate_pixels,
    list_images,
)


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
    def from_tiff_images(cls, name, image_dir, num_images, memmap=False, subsample=10):
        """Loads tiff images into a numpy array.

        Data is assumed to be stored in 16-bit unsigned integers. Frame numbers are
        assumed to be padded with zeros: 00000, 00001, 00002, etc. This is required
        such that Python sorts the images correctly. Frame numbers can start from 0, 1,
        or any other number. Files must have the extension ``.tiff``.

        If memmap is True, the data is not loaded into memory bot a memory mapped file
        on disk is used. The file is named ``$name.npy`` and is placed in the
        `image_dir` folder.

        Args:
            name (str): Movie name.
            image_dir (str): Path of image folder.
            num_images (int): Number of images in the folder.
            memmap (bool): If True, a memory-mapped file is used. (*Default: False*)
            subsample (int): Number of frames to average into a single frame.

        Returns:
            Movie: Movie created from image files.
        """

        images, data_size = cls._get_tiff_images_and_size(image_dir, num_images)

        subsampler = Subsampler(data_size, subsample)

        if memmap:
            memmap_filename = os.path.join(image_dir, name + ".npy")
            data = np.memmap(
                memmap_filename,
                dtype=np.float32,
                mode="w+",
                shape=subsampler.output_shape,
            )
        else:
            data = np.zeros(subsampler.output_shape, np.float32)

        cls._read_images(images, data, subsampler)

        return cls(name, data)

    @staticmethod
    def _get_tiff_images_and_size(image_dir, num_images):
        """ Provides a sorted list of images and computes the required array size.

        Data is assumed to be stored in 16-bit unsigned integers. Frame numbers are
        assumed to be padded with zeros: 00000, 00001, 00002, etc. This is required
        such that Python sorts the images correctly. Frame numbers can start from 0, 1,
        or any other number. Files must have the extension ``.tiff``.

        Args:
            image_dir (str): Path of image folder.
            num_images (int): Number of images in the folder.

        Returns:
            tuple[List[Str], tuple]: Tuple of the list of images and the array size.
        """
        images = list_images(image_dir)

        assert len(images) == num_images

        # read image meta data
        first_image = images[0]
        with Image.open(first_image) as image:
            meta = {TAGS[key]: image.tag[key] for key in image.tag}

        # set size of data
        data_size = (len(images), meta["ImageLength"][0], meta["ImageWidth"][0])

        return images, data_size

    @staticmethod
    def _read_images(images, output_array, subsampler):
        """ Loads images and copies them into the provided array.

        Args:
            images (list[Str]): Sorted list image paths.
            output_array (np.array like): T x N_1 x N_2 array-like object into which
                images should be loaded. T must equal the number of images in `images`.
                Each image should be of size N_1 x N_2.
            subsampler

        Returns:
            np.array like: The input array `array`.

        """
        for filename in images:
            if subsampler.buffer_full:
                output_array[
                    slice(*subsampler.buffer_indices), :, :
                ] = subsampler.buffer
                subsampler.advance_buffer()

            with Image.open(filename) as image:
                subsampler.add_frame(np.array(image))

        output_array[slice(*subsampler.buffer_indices), :, :] = subsampler.buffer

        return output_array

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
        return self._data.__getitem__(key).astype(np.float64)

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


class Patch:
    """Square subregion of Movie.

    Patch limits the data used for the segmentation of a potential cell. Given a center
    seed pixel, Patch defines a square subregion centered on the seed pixel with width
    patch_size. If the square extends outside the movie boundaries, then the subregion
    is shifted such that it stays within the movie boundaries.

    The patch also provides an alternative coordinate system with respect to the top
    left pixel of the patch. This pixel is the zero coordinate for the patch coordinate
    system. The coordinate offset is the coordinate of the top left pixel in the movie
    coordinate system.

    Attributes:
        _center_seed (tuple): Seed pixel that marks the potential cell. The pixel is
            represented as a tuple of coordinates. The coordinates are relative to the
            movie. The top left pixel of the movie represents zero.
        _coordinate_offset (tuple): Movie coordinates of the pixel that represents the
            zero coordinate in the Patch object. Similar to the Movie, pixels in the
            Patch are indexed from the top left corner.
        _data (np.array): Subset of the Movie data. Only data for the patch is stored.
        _movie (Movie): Movie for which the Patch object is a subregion.
        _num_dimensions (int): Dimension of the patch. It matches the dimension of the
            movie.
        _patch_size (int): length of the patch in each dimension. Must be an odd number.
    """

    def __init__(self, movie, center_seed, patch_size):
        """Initializes Patch object."""
        if patch_size % 2 == 0:
            raise ValueError("patch_size (%d) should be an odd number.")

        self._num_dimensions = movie.num_dimensions
        self._center_seed = center_seed
        self._patch_size = patch_size
        self._movie = movie
        self._coordinate_offset = self._compute_coordinate_offset()
        self._data = self._movie[self._movie_indices()]

    @property
    def num_frames(self):
        """Number of frames in the Movie."""
        return self._movie.num_frames

    @property
    def pixel_shape(self):
        """Shape of the patch in pixels. Does not not included the time dimension."""
        return (self._patch_size,) * self._num_dimensions

    def _compute_coordinate_offset(self):
        """Computes the coordinate offset of the patch.

        Confirms that the patch falls within the movie boundaries and shifts the patch
        if necessary. The center seed pixel may not be in the center of the patch if a
        shift is necessary.
        """
        half_width = int((self._patch_size - 1) / 2)

        topleft_coordinates = add_offset_to_coordinate(
            self._center_seed, (-half_width,) * self._num_dimensions
        )
        # shift left such that top left corner exists
        topleft_coordinates = list(max(x, 0) for x in topleft_coordinates)

        # bottomright corners (python-style index so not included)
        bottomright_coordinates = add_offset_to_coordinate(
            topleft_coordinates, (self._patch_size,) * self._num_dimensions
        )
        # shift right such that bottom right corner exists
        bottomright_coordinates = list(
            min(x, max_value)
            for x, max_value in zip(bottomright_coordinates, self._movie.pixel_shape)
        )

        topleft_coordinates = add_offset_to_coordinate(
            bottomright_coordinates, (-self._patch_size,) * self._num_dimensions
        )

        return topleft_coordinates

    def _movie_indices(self):
        """Computes the indices of the movie that correspond to the patch.

        For a patch with top left pixel (5, 5) and bottom right pixel (9, 9), this
        method returns ``(:, 5:10, 5:10)`` which can be used to acccess the data
        corresponding to the patch in the movie.
        """
        bottomright_coordinates = add_offset_to_coordinate(
            self._coordinate_offset, (self._patch_size,) * self._num_dimensions
        )

        # pixel indices
        idx = []
        for start, stop in zip(self._coordinate_offset, bottomright_coordinates):
            idx.append(slice(start, stop))
        return add_time_index(tuple(idx))

    def to_movie_coordinate(self, patch_coordinate):
        """Converts a movie coordinate into a patch coordinate.

        Args:
            patch_coordinate (tuple): Coordinates of a pixel in patch coordinate system.

        Returns:
            tuple: Coordinate of pixel in movie coordinate system.
        """
        return add_offset_to_coordinate(patch_coordinate, self._coordinate_offset)

    def to_patch_coordinate(self, movie_coordinate):
        """Converts a movie coordinate into a patch coordinate.

        Args:
            movie_coordinate (tuple): Coordinates of a pixel in movie coordinate system.

        Returns:
            tuple: Coordinate of pixel in patch coordinate system.
        """
        return add_offset_to_coordinate(
            movie_coordinate, [-x for x in self._coordinate_offset]
        )

    def enumerate_pixels(self):
        """Returns the movie coordinates of the pixels in the patch."""
        return add_offset_set_coordinates(
            generate_pixels(self.pixel_shape), self._coordinate_offset
        )

    def __getitem__(self, key):
        """Access data for pixels in the patch. Indexed in patch coordinates."""
        return self._data[key]


class Subsampler:
    """Subsampler for averaging frames.

    Averages `subsample_frequency` into a single frame. Stores averaged frames in a
    buffer and writes buffer to an output array.

    Attributes:
        _buffer (np.array): (b, N_1, N_2) array where the frame averages are compiled.
        _buffer_frame_count: (b, ) array with the number of frames used in each
            averaged frame.
        _buffer_size (int): Number of averaged frames to store in buffer. Short: b.
            Default is 10.
        _buffer_start_index (int): Index of averaged movie corresponding with first
            frame in the buffer.
        _current_index (int): Index of current frame in buffer.
        _movie_shape (int): Shape of input movie.
        _num_effective_frames (int): Number of frames in the averaged movie.
        _subsample_frequency (int): Number of frames to average into a single frame.
    """

    def __init__(self, movie_shape, subsample_frequency, buffer_size=10):
        """Initializes a subsampler object."""
        self._movie_shape = movie_shape
        self._subsample_frequency = subsample_frequency
        self._num_effective_frames = int(
            math.ceil(self._movie_shape[0] / float(self._subsample_frequency))
        )
        self._buffer_size = min(buffer_size, self._num_effective_frames)
        self._current_index = 0

        self._buffer = np.zeros(
            (self._buffer_size, *self._movie_shape[1:]), dtype=np.float32
        )

        self._buffer_frame_count = np.zeros((self._buffer_size,))
        self._buffer_start_index = 0

    @property
    def output_shape(self):
        """Shape of average movie array."""
        return (self._num_effective_frames, *self._movie_shape[1:])

    @property
    def buffer(self):
        """Provides access to data in buffer. Corrects last buffer for movie length."""
        # x % 10 takes values in 0...9 whereas x - 1 % 10 + 1 takes values 1, .. 10
        max_index = ((self.buffer_indices[1] - 1) % self._buffer_size) + 1
        return self._buffer[:max_index, :, :]

    @property
    def buffer_full(self):
        """True if buffer is full."""
        return self._current_index >= self._buffer_size

    @property
    def buffer_indices(self):
        """Indices in average movie corresponding to current buffer"""
        return (
            self._buffer_start_index,
            min(
                self._buffer_start_index + self._buffer_size, self._num_effective_frames
            ),
        )

    def add_frame(self, frame):
        """ Adds frame to average.

        Frames should be provided in order of appearance in the movie.

        Args:
            frame (np.array): (N_1, N_2) array with pixel intensities.

        Returns:
            None

        Raises:
            ValueError: If buffer is full.
        """
        if self.buffer_full:
            raise ValueError("Buffer is full. Cannot add current frame.")

        current_num_frames = self._buffer_frame_count[self._current_index]
        new_num_frames = float(current_num_frames + 1)
        self._buffer[self._current_index, :, :] = (
            current_num_frames
            / new_num_frames
            * self._buffer[self._current_index, :, :]
            + frame / new_num_frames
        )
        self._buffer_frame_count[self._current_index] += 1

        # advance to next frame in buffer if current frame is full
        if self._buffer_frame_count[self._current_index] == self._subsample_frequency:
            self._current_index += 1

    def advance_buffer(self):
        """Empties buffer and advances the buffer indices for new frames"""
        self._buffer_frame_count[:] = 0
        self._buffer[:] = 0
        self._current_index = 0
        self._buffer_start_index += self._buffer_size
