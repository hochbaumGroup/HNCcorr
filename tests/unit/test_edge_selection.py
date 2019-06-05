import numpy as np
import pytest

from hnccorr.edge_selection import SparseComputation


@pytest.fixture
def embedding():
    class MockEmbedding:
        def __init__(self):
            self.embedding = np.array([[-1, 0], [-0.9, 0], [0, 0], [0.9, 0], [1, 0]]).T

    return MockEmbedding()


def test_sparse_computation(embedding):
    assert SparseComputation(2, 0.2).select_edges(embedding) == {
        ((0,), (1,)),
        ((3,), (4,)),
    }


def test_sparse_computation_with_dimension_reducer(embedding):
    class MockDimReducer:
        def fit_transform(self, data, **kwargs):
            return data

    assert SparseComputation(2, 0.2, dimension_reducer=MockDimReducer()).select_edges(
        embedding
    ) == {((0,), (1,)), ((3,), (4,))}
