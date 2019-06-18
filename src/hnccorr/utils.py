import glob
import os
from itertools import product


def add_offset_set_coordinates(iterable, offset):
    """Add offset to each coordinate in set"""
    return set(add_offset_coordinates(c, offset) for c in iterable)


def add_offset_coordinates(coordinates, offset):
    """Add offset"""
    return tuple(a + b for a, b in zip(coordinates, offset))


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
