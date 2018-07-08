from copy import deepcopy
from closure.hnc import HNC as HNC_Closure

from hnccorr.segmentation import Segmentation


class HNC:
    def __init__(self, patch, graph, arc_weight):
        self._patch = patch

        self._hnc = HNC_Closure(
            deepcopy(graph),
            self._patch.positive_seeds,
            self._patch.negative_seeds,
            arc_weight=arc_weight,
        )

    def _construct_segmentations(self, cuts, breakpoints):
        return [
            Segmentation(self._patch, selection, weight)
            for selection, weight in zip(cuts, breakpoints)
        ]

    def solve_parametric(self, low, high):
        print(self._patch.negative_seeds)
        cuts, breakpoints = self._hnc.solve_parametric(low, high)
        return self._construct_segmentations(cuts, breakpoints)
