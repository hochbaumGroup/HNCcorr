import networkx as nx


class GraphConstructor:
    """Graph constructor over a set of pixels.

    Constructs a similarity graph over the set of pixels in a patch. Edges are selected
    by an edge_selector and the similarity weight associated with each edge is computed
    with the weight_function. Edge weights are stored under the attribute ``weight``.

    Attributes:
        _edge_selector (EdgeSelector): Object that constructs the edge set of the graph.
        _weight_function (function): Function that computes the edge weight between two
            pixels. The function should take as input two 1-dimensional numpy arrays,
            representing the feature vectors of the two pixels. The function should
            return a float between 0 and 1.
    """

    def __init__(self, edge_selector, weight_function):
        """Initializes a graph constructor."""
        self._edge_selector = edge_selector
        self._weight_function = weight_function

    def construct(self, patch, embedding):
        """Constructs similarity graph for a given patch.

        See class description.

        Args:
            patch (Patch): Defines subregion and pixel set for the graph.
            embedding (CorrelationEmbedding): Provides feature vectors associated with
                each pixel in the patch.

        Returns:
            nx.Graph: Similarity graph over pixels in patch.
        """
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
