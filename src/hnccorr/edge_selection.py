import numpy as np
from sparsecomputation import SparseComputation as SC
from sparsecomputation import ApproximatePCA


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
        self._dim_low = int(dim_low)
        self._distance = distance

        if dimension_reducer is None:
            self._dim_reducer = ApproximatePCA(self._dim_low)
        else:
            self._dim_reducer = dimension_reducer

        self._sc = SC(self._dim_reducer, distance=self._distance)

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
