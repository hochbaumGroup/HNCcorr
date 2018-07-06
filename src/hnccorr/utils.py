import glob
import os
from itertools import product
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes
import networkx as nx


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
    def neighborhood(num_dims):
        for dim, change in product(range(num_dims), (-1, 1)):
            shift = [0] * num_dims
            shift[dim] = change
            yield shift

    neighbors = tuple(neighborhood(num_dims))

    graph = nx.Graph()
    graph.add_nodes_from(selection)

    for index, shift in product(selection, neighbors):
        neighbor = tuple(map(lambda a, b: a + b, index, shift))
        if neighbor in graph:
            graph.add_edge(index, neighbor)

    components = list(nx.connected_components(graph))

    overlap = [len(c.intersection(seeds)) for c in components]

    best_component = components[np.argmax(overlap)]

    return best_component


def eight_neighborhood(num_dims, max_radius):
    return set(product(range(-max_radius, max_radius + 1), repeat=num_dims))
