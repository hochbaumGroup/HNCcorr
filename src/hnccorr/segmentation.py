import networkx as nx
import numpy as np
from itertools import product
from scipy.ndimage.morphology import binary_fill_holes

from hnccorr.utils import four_neighborhood


class Segmentation:
    def __init__(self, selection, weight):
        self.selection = set(selection)
        self.weight = weight

    def __eq__(self, other):
        return (self.selection == other.selection) and (self.weight == other.weight)

    def clean(self, positive_seeds, region_size):
        """Remove left over points / fill holes"""
        improved_segmentation = self._select_max_seed_component(positive_seeds)
        return improved_segmentation.fill_holes(region_size)

    def _select_max_seed_component(self, seeds):
        # get an arbitrary element from seeds to compute dimension
        num_dims = len(next(iter(self.selection)))
        neighbors = four_neighborhood(num_dims)

        graph = nx.Graph()
        graph.add_nodes_from(self.selection)

        for index, shift in product(self.selection, neighbors):
            neighbor = tuple(map(lambda a, b: a + b, index, shift))
            if neighbor in graph:
                graph.add_edge(index, neighbor)

        components = list(nx.connected_components(graph))

        overlap = [len(c.intersection(seeds)) for c in components]

        best_component = components[np.argmax(overlap)]

        return Segmentation(best_component, self.weight)

    def fill_holes(self, patch_shape):
        mask = np.full(patch_shape, False, dtype=np.bool)

        indices = list(zip(*self.selection))
        mask[indices] = True

        filled_mask = binary_fill_holes(mask)

        index_arrays = [a.tolist() for a in np.where(filled_mask)]
        return Segmentation(set(zip(*index_arrays)), self.weight)
