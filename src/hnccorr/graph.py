import networkx as nx

from hnccorr.utils import generate_pixels


class GraphConstructor(object):
    def __init__(self, patch, edge_selector, weight_function):
        self._patch = patch
        self._edge_selector = edge_selector
        self._weight_function = weight_function
        self.arc_weight = "weight"

    def construct(self):
        G = nx.Graph()

        G.add_nodes_from(generate_pixels(self._patch.pixel_size))

        for a, b in self._edge_selector.select_edges():
            G.add_edge(a, b, weight=self._weight_function(a, b))

        return G
