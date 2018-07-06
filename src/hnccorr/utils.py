import glob
import os
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes


def add_offset_set_coordinates(x, offset):
    """Add offset to each coordinate in set"""
    return set(add_offset_coordinates(c, offset) for c in x)


def add_offset_coordinates(x, offset):
    """Add offset"""
    return tuple(a + b for a, b in zip(x, offset))


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
    files_tif = glob.glob(os.path.join(folder, "*.tif"))
    return sorted(files_tif)


def fill_holes(selection, patch_shape):
    mask = np.full(patch_shape, False, dtype=np.bool)

    indices = list(zip(*selection))
    mask[indices] = True

    filled_mask = binary_fill_holes(mask)

    index_arrays = [a.tolist() for a in np.where(filled_mask)]
    return set(zip(*index_arrays))
