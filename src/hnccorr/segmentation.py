import networkx as nx
import numpy as np
from itertools import product

from hnccorr.utils import fill_holes, four_neighborhood


class Segmentation:
    def __init__(self, selection, weight):
        self.selection = set(selection)
        self.weight = weight

    def __eq__(self, other):
        return (self.selection == other.selection) and (self.weight == other.weight)

    def clean(self, positive_seeds, region_size):
        """Remove left over points / fill holes"""
        new_segmentation = self.select_max_seed_component(
            self.selection, positive_seeds
        )
        self.selection = fill_holes(new_segmentation.selection, region_size)
        return self

    def select_max_seed_component(self, selection, seeds):
        # get an arbitrary element from seeds to compute dimension
        num_dims = len(next(iter(selection)))
        neighbors = four_neighborhood(num_dims)

        graph = nx.Graph()
        graph.add_nodes_from(selection)

        for index, shift in product(selection, neighbors):
            neighbor = tuple(map(lambda a, b: a + b, index, shift))
            if neighbor in graph:
                graph.add_edge(index, neighbor)

        components = list(nx.connected_components(graph))

        overlap = [len(c.intersection(seeds)) for c in components]

        best_component = components[np.argmax(overlap)]

        return Segmentation(best_component, self.weight)
