import networkx as nx
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes
from itertools import product


class Segmentation(object):
    def __init__(self, patch, selection, weight):
        self._patch = patch
        self.selection = set(selection)
        self.weight = weight

    def _fill_holes(self):
        shape = self._patch.shape
        mask = np.full(shape, False, dtype=np.bool)

        indices = list(zip(*self.selection))
        mask[indices] = True

        filled_mask = binary_fill_holes(mask)

        index_arrays = [a.tolist() for a in np.where(filled_mask)]
        self.selection = set(zip(*index_arrays))

    def _select_largest_component(self):
        shape = self._patch.shape
        n = np.product(shape)
        zero_tuple = (0,) * len(shape)

        def tuple_add(x, y):
            return tuple(i + j for i, j in zip(x, y))

        max_tuple = tuple_add(shape, (-1,) * len(shape))

        def neighborhood(num_dims):
            for dim, change in product(range(num_dims), (-1, 1)):
                shift = [0] * num_dims
                shift[dim] = change
                yield shift

        neighbors = tuple(neighborhood(len(shape)))

        G = nx.Graph()
        G.add_nodes_from(self.selection)

        for index, shift in product(self.selection, neighbors):
            neighbor = tuple_add(index, shift)
            if zero_tuple <= neighbor <= max_tuple and neighbor in G:
                G.add_edge(index, neighbor)

        components = list(nx.connected_components(G))

        overlap = [
            len(c.intersection(self._patch.positive_seeds)) for c in components
        ]

        best_component = components[np.argmax(overlap)]

        self.selection = best_component

    def clean(self):
        """Remove left over points / fill holes"""
        self._select_largest_component()
        self._fill_holes()
