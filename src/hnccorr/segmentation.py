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
"""HNC and segmentation related components in HNCcorr."""

from copy import deepcopy
from itertools import product
import networkx as nx
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes

from closure.hnc import HNC as HNC_Closure

from hnccorr.utils import four_neighborhood


class HncParametricWrapper:
    r"""Wrapper for solving the Hochbaum Normalized Cut (HNC) problem on a graph.

    Given an undirected graph :math:`G = (V, E)` with edge weights :math:`w_{ij} \ge 0`
    for :math:`[i,j] \in E`, the linearized HNC problem is defined as:

    .. math::
        \min_{\emptyset \subset S \subset V} \sum_{\substack{[i,j] \in E,\\ i \in S,\\
        j \in V \setminus S}} w_{ij} - \lambda \sum_{i \in S} d_i,

    where $d_i$ the degree of node :math:`i \in V` and :math:`\lambda \ge 0` provides
    the trade-off between the two objective terms.

    See closure package for solution method.
    """

    def __init__(self, lower_bound, upper_bound):
        """Initializes HncParametricWrapper object."""
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    @staticmethod
    def _construct_segmentations(source_sets, breakpoints):
        """Constructs a list of segmentations from output HNC.

        Each source set and corresponding lambda upper bound is replaced with a
        Segmentation object where the selection matches the source set and the weight
        parameter matches the upper bound of the lambda range.

        Args:
            source_sets (list[set]): List of source sets for each lambda range.
            breakpoints (list[float]): List of upper bounds on the lambda range for
                which the corresponding source set is optimal.

        Returns:
            list[Segmentation]: List of segmentations.

        """
        return [
            Segmentation(selection, weight)
            for selection, weight in zip(source_sets, breakpoints)
        ]

    def solve(self, graph, pos_seeds, neg_seeds):
        """Solves an instance of the HNC problem for all values of lambda.

        Solves the HNC clustering problem on `graph` for all values of lambda
        simultaneously. See class description for a definition of HNC.

        Args:
            graph (nx.Graph): Directed similarity graph with non-negative edge
                weights. Edge [i,j] is represented by two directed arcs (i,j) and
                (j,i). Edge weights must be defined via the attribute `weight`.
            pos_seeds (set): Set of nodes in graph that must be part of the cluster.
            neg_seeds (set): Set of nodes in graph that must be part of the complement.

        Returns:
            list[Segmentation]: List of optimal clusters for each lambda range.

        Caution:
            Class modifies graph for performance. Pass a copy to prevent any issues.
        """
        hnc = HNC_Closure(graph, pos_seeds, neg_seeds, arc_weight="weight")
        source_sets, breakpoints = hnc.solve_parametric(
            self._lower_bound, self._upper_bound
        )
        return self._construct_segmentations(source_sets, breakpoints)


class Segmentation:
    """A set of pixels identified by HNC as a potential cell footprint.

    Attributes:
        selection (set): Pixels in the spatial footprint. Each pixel is represented as
            a tuple.
        weight (float): Upper bound on the lambda coefficient for which this
            segmentation is optimal.
    """

    def __init__(self, selection, weight):
        """Initializes a Segmentation object."""
        self.selection = set(selection)
        self.weight = weight

    def __eq__(self, other):
        """Compares two Segmentation objects."""
        if isinstance(other, Segmentation):
            return (self.selection == other.selection) and (self.weight == other.weight)

        return False

    def clean(self, positive_seeds, movie_pixel_shape):
        """Cleans Segmentation by selecting a connected component and filling holes.

        The Segmentation is decomposed into connected components by considering
        horizontal or vertical adjacent pixels as neighbors. The connected component
        with the most positive seeds is selected. Any holes in the selected component
        are added to the selection.

        Args:
            positive_seeds (set): Pixels that are contained in the
                spatial footprint. Each pixel is represented by a tuple.
            movie_pixel_shape (tuple): Pixel resolution of the movie.

        Returns:
            Segmentation: A new Segmentation object with the same weight.
        """
        improved_segmentation = self.select_max_seed_component(positive_seeds)
        return improved_segmentation.fill_holes(movie_pixel_shape)

    def select_max_seed_component(self, positive_seeds):
        """Selects the connected component of selection that contains the most seeds.

        The Segmentation is decomposed into connected components by considering
        horizontal or vertical adjacent pixels as neighbors. The connected component
        with the most positive seeds is selected.

        Args:
            positive_seeds (set): Pixels that are contained in the
                spatial footprint. Each pixel is represented by a tuple.

        Returns:
            Segmentation: A new Segmentation object with the same weight.
        """

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

        overlap = [len(c.intersection(positive_seeds)) for c in components]

        best_component = components[np.argmax(overlap)]

        return Segmentation(best_component, self.weight)

    def fill_holes(self, movie_pixel_shape):
        """Fills holes in the selection.

        Args:
            movie_pixel_shape (tuple): Pixel resolution of the movie.

        Returns:
            Segmentation: A new Segmentation object with the same weight.
        """
        mask = np.full(movie_pixel_shape, False, dtype=np.bool)

        indices = tuple(zip(*self.selection))
        mask[indices] = True

        filled_mask = binary_fill_holes(mask)

        index_arrays = [a.tolist() for a in np.where(filled_mask)]
        return Segmentation(set(zip(*index_arrays)), self.weight)
