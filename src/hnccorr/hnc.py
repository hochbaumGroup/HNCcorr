from copy import deepcopy
from closure.hnc import HNC as HNC_Closure

from hnccorr.segmentation import Segmentation


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
            graph (nx.Graph): Undirected similarity graph with non-negative edge
                weights. Edge weights must be defined via the attribute `weight`.
            pos_seeds (set): Set of nodes in graph that must be part of the cluster.
            neg_seeds (set): Set of nodes in graph that must be part of the complement.

        Returns:
            list[Segmentation]: List of optimal clusters for each lambda range.
        """
        hnc = HNC_Closure(deepcopy(graph), pos_seeds, neg_seeds, arc_weight="weight")
        source_sets, breakpoints = hnc.solve_parametric(
            self._lower_bound, self._upper_bound
        )
        return self._construct_segmentations(source_sets, breakpoints)
