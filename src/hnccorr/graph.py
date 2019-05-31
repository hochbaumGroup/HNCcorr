import networkx as nx

from hnccorr.utils import generate_pixels


class GraphConstructor:
    def __init__(self, edge_selector, weight_function):
        self._edge_selector = edge_selector
        self._weight_function = weight_function
        self.arc_weight = "weight"

    def construct(self, patch, embedding):
        graph = nx.Graph()

        graph.add_nodes_from(patch.enumerate_pixels())

        for node1, node2 in self._edge_selector.select_edges(embedding):
            graph.add_edge(node1, node2, weight=self._weight_function(node1, node2))

        return graph
