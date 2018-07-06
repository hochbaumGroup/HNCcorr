import glob
import os
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes
import networkx as nx
from itertools import product


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


def select_max_seed_component(selection, seeds, num_dims):
    def tuple_add(x, y):
        return tuple(i + j for i, j in zip(x, y))

    def neighborhood(num_dims):
        for dim, change in product(range(num_dims), (-1, 1)):
            shift = [0] * num_dims
            shift[dim] = change
            yield shift

    neighbors = tuple(neighborhood(num_dims))

    G = nx.Graph()
    G.add_nodes_from(selection)

    for index, shift in product(selection, neighbors):
        neighbor = tuple_add(index, shift)
        if neighbor in G:
            G.add_edge(index, neighbor)

    components = list(nx.connected_components(G))

    overlap = [len(c.intersection(seeds)) for c in components]

    best_component = components[np.argmax(overlap)]

    return best_component
