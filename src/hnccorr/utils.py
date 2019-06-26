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
"""Helper functions for HNCcorr."""

import glob
import os
from itertools import product


def add_offset_set_coordinates(iterable, offset):
    """Adds a fixed offset to all pixel coordinates in a set.

    Args:
        coordinates (set): Set of pixel coordinates. Each pixel coordinate is a tuple.
        offset (tuple): Offset to add to each pixel coordinate. Tuple should be of the
            same length as the tuples in `coordinates`.

    Returns:
        set: Set of updated coordinates.

    Example:
        .. code-block:: python

            >>> add_offset_set_coordinates({(5, 2), (4, 7)}, (2, 2))
            {(7, 4), (6, 9)}
    """
    return set(add_offset_to_coordinate(c, offset) for c in iterable)


def add_offset_to_coordinate(coordinate, offset):
    """Offsets pixel coordinate by another coordinate.

    Args:
        coordinate (tuple): Pixel coordinate to offset.
        offset (tuple): Offset to add to coordinate. Must be of the same length.

    Example:
        .. code-block:: python

            >>> add_offset_to_coordinate((5, 3, 4), (1, -1, 3))
            (6, 2, 7)
    """
    return tuple(a + b for a, b in zip(coordinate, offset))


def add_time_index(index):
    """Inserts a full slice as the first dimension of an index for e.g. numpy.

    Args:
        index (tuple): Index for e.g. numpy array.

    Returns:
        tuple: New index with additional dimension.

    Example:
        .. code-block:: python

            >>> add_time_index((5, :3))
            (:, 5, :3)
    """
    return (slice(None, None),) + index


def list_images(folder):
    """Lists and sorts tiff images in a folder.

    Images are sorted in ascending order based on filename.

    Caution:
        Filenames are sorted as strings. Note that ``200.tiff`` is sorted before
        ``5.tiff``. Pad image filenames with zeros to prevent this: ``005.tiff``.

    Args:
        folder: folder containing tiff image files.

    Returns:
        list: Sorted list of paths of tiff files in folder.
    """
    files_tif = glob.glob(os.path.join(folder, "*.tiff"))
    return sorted(files_tif)


def four_neighborhood(num_dims):
    """Returns all neighboring pixels of zero that differ in at most one coordinate.

    Includes zero coordinate itself.

    Args:
        num_dims (int): Number of dimensions for the coordinates.

    Returns:
        set: Set of pixel coordinates.

    Example:
        .. code-block:: python

            >>> four_neighborhood(1)
            [(-1,), (0,), (1,)]
            >>> eight_neighborhood(2)
            [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)]

    """
    neighbors = []
    for dim, change in product(range(num_dims), (-1, 0, 1)):
        shift = [0] * num_dims
        shift[dim] = change
        neighbors.append(tuple(shift))
    return set(neighbors)


def eight_neighborhood(num_dims, max_radius):
    """Returns all coordinates within a given L-infinity distance of zero.

    Includes zero coordinate itself.

    Args:
        num_dims (int): Number of dimensions for the coordinates.
        max_radius (int): Largest L-infinity distance allowed.

    Returns:
        set: Set of pixel coordinates.

    Example:
        .. code-block:: python

            >>> eight_neighborhood(1, 1)
            [(-1,), (0,), (1,)]
            >>> eight_neighborhood(2, 1)
            [
                (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0),
                (0, 1), (1, -1), (1, 0), (1, 1)
            ]

    """
    return set(product(range(-max_radius, max_radius + 1), repeat=num_dims))


def generate_pixels(shape):
    """Enumerate all pixel coordinates for a movie/patch.

    Args:
        shape (tuple): Shape of movie. Number of pixels in each dimension.

    Returns:
        Iterator: Iterates over all pixels.

    Example:
        .. code-block:: python

            >>> generate_pixels((2,2))
            [(0, 0), (0, 1), (1, 0), (1, 1)]
    """
    return product(*[range(n) for n in shape])
