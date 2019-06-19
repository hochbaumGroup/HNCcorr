import glob
import os
from itertools import product


def add_offset_set_coordinates(iterable, offset):
    """Add offset to each coordinate in set"""
    return set(add_offset_coordinates(c, offset) for c in iterable)


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
    """Add full slice to from time dimension to pixel index"""
    return (slice(None, None),) + index


def list_images(folder):
    """List and sort tiff images

    Args:
        folder: folder containing tif(f) files.

    Returns:
        list: Sorted list of paths of tiff files in folder.
    """
    files_tif = glob.glob(os.path.join(folder, "*.tiff"))
    return sorted(files_tif)


def four_neighborhood(num_dims):
    neighbors = []
    for dim, change in product(range(num_dims), (-1, 0, 1)):
        shift = [0] * num_dims
        shift[dim] = change
        neighbors.append(tuple(shift))
    return set(neighbors)


def eight_neighborhood(num_dims, max_radius):
    return set(product(range(-max_radius, max_radius + 1), repeat=num_dims))


def generate_pixels(shape):
    return product(*[range(n) for n in shape])
