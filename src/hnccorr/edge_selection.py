import numpy as np
from sparsecomputation import SparseComputation as SC
from sparsecomputation import ApproximatePCA


class SparseComputation:
    def __init__(self, dim_low, distance):
        self._dim_low = int(dim_low)
        self._distance = distance

        self._apca = ApproximatePCA(self._dim_low)
        self._sc = SC(self._apca, distance=self._distance)

    def select_edges(self, embedding):
        shape = embedding.embedding.shape[1:]
        data = embedding.embedding.reshape(-1, np.product(shape)).T

        pairs = self._sc.select_pairs(data)

        return [
            (np.unravel_index(a, shape), np.unravel_index(b, shape))
            for a, b in pairs
        ]
