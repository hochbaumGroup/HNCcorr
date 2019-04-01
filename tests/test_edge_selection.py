import numpy as np

from hnccorr.edge_selection import SparseComputation


def test_sparse_computation():
    class MockEmbedding:
        def __init__(self):
            self.embedding = np.array(
                [[-1, 0], [-0.9, 0], [0, 0], [0.9, 0], [1, 0]]
            ).T

    embedding = MockEmbedding()

    sc = SparseComputation(2, 0.2)

    assert set(sc.select_edges(embedding)) == {((0,), (1,)), ((3,), (4,))}
