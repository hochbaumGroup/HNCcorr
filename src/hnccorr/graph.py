import networkx as nx

from hnccorr.utils import generate_pixels


class GraphConstructor(object):
    def __init__(self, patch, edge_selector, embedding):
        self._patch = patch
        self._edge_selector = edge_selector
        self._embedding = embedding

    def construct(self):
        G = nx.Graph()

        G.add_nodes_from(generate_pixels(self._patch.pixel_size))

        for a, b in self._edge_selector.select_edges():
            G.add_edge(a, b, weight=self._embedding.distance(a, b))

        return G
