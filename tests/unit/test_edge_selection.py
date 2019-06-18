import numpy as np
import pytest

from hnccorr.edge_selection import SparseComputationEmbeddingWrapper


@pytest.fixture
def mock_embedding(mocker, dummy):
    return mocker.patch("hnccorr.embedding.CorrelationEmbedding", autospec=True)(dummy)


def test_sparse_computation(mock_embedding):
    mock_embedding.embedding = np.array(
        [[-1, 0], [-0.9, 0], [0, 0], [0.9, 0], [1, 0]]
    ).T
    assert SparseComputationEmbeddingWrapper(2, 0.2).select_edges(mock_embedding) == {
        ((0,), (1,)),
        ((3,), (4,)),
    }


def test_sparse_computation_with_dimension_reducer(mock_embedding):
    mock_embedding.embedding = np.array(
        [[-1, 0], [-0.9, 0], [0, 0], [0.9, 0], [1, 0]]
    ).T

    class MockDimReducer:
        def fit_transform(self, data, **kwargs):
            return data

    assert SparseComputationEmbeddingWrapper(
        2, 0.2, dimension_reducer=MockDimReducer()
    ).select_edges(mock_embedding) == {((0,), (1,)), ((3,), (4,))}
