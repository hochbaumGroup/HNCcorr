"""HNCcorr components related to the similarity graph."""

import networkx as nx
import numpy as np
from sparsecomputation import SparseComputation as SC
from sparsecomputation import ApproximatePCA

from hnccorr.utils import add_time_index


class CorrelationEmbedding:
    """Represents each pixel as a vector of correlations to other pixels."""

    def __init__(self, patch):
        data = patch[:].reshape(-1, np.product(patch.pixel_shape))
        self.embedding = np.corrcoef(data.T).reshape(-1, *patch.pixel_shape)
        self.embedding[np.isnan(self.embedding)] = 0
        self._length = self.embedding.shape[0]

    def get_vector(self, pixel):
        return self.embedding[add_time_index(pixel)]


def exponential_distance_decay(feature_vec1, feature_vec2, alpha):
    return np.exp(-alpha * np.mean(np.power(feature_vec1 - feature_vec2, 2)))


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


class SparseComputationEmbeddingWrapper:
    """Wrapper for SparseComputation that accepts an embedding.

    Attributes:
        _sc (SparseComputation): SparseComputation object.
    """

    def __init__(self, dim_low, distance, dimension_reducer=None):
        """Initializes a SparseComputationEmbeddingWrapper instance.

        Args:
            dim_low (int): Dimension of the low-dimensional space in sparse computation.
            distance (float): 1 / grid_resolution. Defines the size of the grid blocks
                in sparse computation.
            dimension_reducer (DimReducer): Provides dimension reduction for sparse
                computation. By default, approximate principle component analysis is
                used.

        Returns:
            SparseComputationEmbeddingWrapper

        """
        if dimension_reducer is None:
            dimension_reducer = ApproximatePCA(int(dim_low))

        self._sc = SC(dimension_reducer, distance=distance)

    def select_edges(self, embedding):
        """Selects relevant pairwise similarities with sparse computation.

        Determines the set of relevant pairwise similarities based on the sparse
        computation algorithm. See sparse computation for details. Pixel coordinates
        are with respect to the index of the embedding.

        Args:
            embedding (CorrelationEmbedding): Embedding of pixels into feature vectors.

        Returns:
            list(tuple): List of relevant pixel pairs.
        """
        shape = embedding.embedding.shape[1:]
        data = embedding.embedding.reshape(-1, np.product(shape)).T

        pairs = self._sc.select_pairs(data)

        return set(
            (np.unravel_index(a, shape), np.unravel_index(b, shape)) for a, b in pairs
        )
