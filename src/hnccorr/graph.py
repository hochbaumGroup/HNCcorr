# Copyright © 2017. Regents of the University of California (Regents). All Rights
# Reserved.
#
# Permission to use, copy, modify, and distribute this software and its documentation
# for educational, research, and not-for-profit purposes, without fee and without a
# signed licensing agreement, is hereby granted, provided that the above copyright
# notice, this paragraph and the following two paragraphs appear in all copies,
# modifications, and distributions. Contact The Office of Technology Licensing, UC
# Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201,
# for commercial licensing opportunities. Created by Quico Spaen, Roberto Asín-Achá,
# and Dorit S. Hochbaum, Department of Industrial Engineering and Operations Research,
# University of California, Berkeley.
#
# IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
# INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE
# OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE
# SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS
# IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES,
# ENHANCEMENTS, OR MODIFICATIONS.
"""HNCcorr components related to the similarity graph."""

import networkx as nx
import numpy as np
from sparsecomputation import SparseComputation as SC
from sparsecomputation import ApproximatePCA

from hnccorr.utils import add_time_index


class CorrelationEmbedding:
    """Computes correlation feature vector for each pixel.

    Embedding provides a representation of a pixel in terms of feature vector. The
    feature vector for the CorrelationEmbedding is a vector of pairwise correlations to
    each (or some) pixel in the patch.

    If the correlation is not defined due to a pixel with zero variance, then the
    corelation is set to zero.


    Attributes:
        embedding (np.array): (D, N_1, N_2, ..) array of pairwise correlations, where D
            is the dimension of the embedding and N_1, N_2, .. are the pixel shape of
            the patch.
    """

    def __init__(self, patch):
        """Initializes a CorrelationEmbedding object.

        See class description for details.

        Args:
            patch (Patch): Subregion of movie for which the correlation embedding is
                computed.
        """
        data = patch[:].reshape(-1, np.product(patch.pixel_shape))
        self.embedding = np.corrcoef(data.T).reshape(-1, *patch.pixel_shape)
        self.embedding[np.isnan(self.embedding)] = 0

    def get_vector(self, pixel):
        """Retrieve feature vector of pixel.

        Args:
            pixel (tuple): Coordinate of pixel.

        Returns:
            np.array: Feature vector of pixel.
        """
        return self.embedding[add_time_index(pixel)]


def exponential_distance_decay(feature_vec1, feature_vec2, alpha):
    """Computes ``exp(- alpha / n || x_1 - x_2 ||^2_2)`` for x_1, x_2 in R^n."""
    num_frames = float(feature_vec1.shape[0])
    return np.exp(
        -alpha * np.linalg.norm(feature_vec1 - feature_vec2) ** 2 / num_frames
    )


class GraphConstructor:
    """Graph constructor over a set of pixels.

    Constructs a similarity graph over the set of pixels in a patch. Edges are selected
    by an edge_selector and the similarity weight associated with each edge is computed
    with the weight_function. Edge weights are stored under the attribute ``weight``.

    A directed graph is used for efficiency. That is, arcs (i,j) and (j,i) are used to represent edge [i,j].

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
            nx.DiGraph: Similarity graph over pixels in patch.
        """
        graph = nx.DiGraph()

        graph.add_nodes_from(patch.enumerate_pixels())

        for node1, node2 in self._edge_selector.select_edges(embedding):
            node1_movie = patch.to_movie_coordinate(node1)
            node2_movie = patch.to_movie_coordinate(node2)
            weight = self._weight_function(
                embedding.get_vector(node1), embedding.get_vector(node2)
            )
            # add arc in both directions
            graph.add_edge(node1_movie, node2_movie, weight=weight)
            graph.add_edge(node2_movie, node1_movie, weight=weight)

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
