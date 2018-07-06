import glob
import os


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
