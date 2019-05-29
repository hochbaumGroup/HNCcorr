from copy import deepcopy
from closure.hnc import HNC as HNC_Closure

from hnccorr.segmentation import Segmentation


class HncParametric:
    def __init__(self, lower_bound, upper_bound):
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    def _construct_segmentations(self, cuts, breakpoints):
        return [
            Segmentation(selection, weight)
            for selection, weight in zip(cuts, breakpoints)
        ]

    def solve(self, seeds, graph):
        hnc = HNC_Closure(
            deepcopy(graph),
            seeds.positive_seeds,
            seeds.negative_seeds,
            arc_weight="weight",
        )
        cuts, breakpoints = hnc.solve_parametric(self._lower_bound, self._upper_bound)
        return self._construct_segmentations(cuts, breakpoints)
