import networkx as nx


class GraphConstructor:
    def __init__(self, edge_selector, weight_function):
        self._edge_selector = edge_selector
        self._weight_function = weight_function
        self.arc_weight = "weight"

    def construct(self, patch, embedding):
        graph = nx.Graph()

        graph.add_nodes_from(patch.enumerate_pixels())

        for node1, node2 in self._edge_selector.select_edges(embedding):
            graph.add_edge(
                patch.to_movie_coordinate(node1),
                patch.to_movie_coordinate(node2),
                weight=self._weight_function(
                    embedding.get_vector(node1), embedding.get_vector(node2)
                ),
            )

        return graph
