"""HNCcorr components related to the similarity graph."""

import pytest
import numpy as np
from hnccorr.graph import (
    CorrelationEmbedding,
    exponential_distance_decay,
    GraphConstructor,
    SparseComputationEmbeddingWrapper,
)


@pytest.fixture
def mock_patch(mocker, dummy):
    return mocker.patch("hnccorr.movie.Patch", autospec=True)(dummy, dummy, dummy)


@pytest.fixture
def mock_edge_selector(mocker, dummy):
    return mocker.patch(
        "hnccorr.graph.SparseComputationEmbeddingWrapper", autospec=True
    )(dummy, dummy)


@pytest.fixture
def mock_embedding(mocker, dummy):
    return mocker.patch("hnccorr.graph.CorrelationEmbedding", autospec=True)(dummy)


@pytest.fixture
def CE1(mock_patch):
    mock_patch.pixel_shape = (7,)
    mock_patch.__getitem__.return_value = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [-1, -1, -1, -1, -1, -1, -1],
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        ]
    )

    return CorrelationEmbedding(mock_patch)


@pytest.fixture
def CE2(mock_patch):
    mock_patch.pixel_shape = (3, 3)
    mock_patch.__getitem__.return_value = np.zeros((3, 3))
    return CorrelationEmbedding(mock_patch)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_embedding_embedding(CE1, CE2, mock_patch):

    np.testing.assert_allclose(
        CE1.embedding[0],
        [1.0, 0.99833749, 0.99339927, 0.98532928, 0.9743547, 0.96076892, 0.94491118],
    )

    np.testing.assert_allclose(CE2.embedding[(0, 0, slice(None, None))], [0, 0, 0])


def test_embedding_get_vector(CE1):
    CE1.embedding = np.array([[0.0, 1.0], [-2.0, 0.0]])
    np.testing.assert_allclose(CE1.get_vector((0,)), np.array([0.0, -2.0]))


def test_exponential_distance_decay():
    alpha = 0.5

    exponential_distance_decay(
        np.array([0.0, -2.0]), np.array([[1.0, 0.0]]), alpha
    ) == pytest.approx(np.exp(-0.25))


def test_graph_constructor(mock_patch, mock_edge_selector, mock_embedding):
    mock_edge_selector.select_edges.return_value = [((0,), (1,)), ((0,), (2,))]
    mock_patch.enumerate_pixels.return_value = [(i,) for i in range(7)]
    mock_patch.to_movie_coordinate = lambda x: x
    mock_embedding.get_vector = lambda x: x

    GC = GraphConstructor(mock_edge_selector, lambda a, b: b[0])
    G = GC.construct(mock_patch, mock_embedding)

    num_nodes = 7
    assert len(G.nodes) == num_nodes
    for i in range(num_nodes):
        assert (i,) in G.nodes

    assert len(G.edges) == 2
    assert ((0,), (1,)) in G.edges
    assert ((0,), (2,)) in G.edges
    assert G[(0,)][(1,)]["weight"] == 1
    assert G[(0,)][(2,)]["weight"] == 2


def test_graph_constructor_nodes_offset_from_zero(
    mock_patch, mock_edge_selector, mock_embedding
):
    all_pixels = {(2,), (3,), (4,), (5,), (6,), (7,), (8,)}
    mock_patch.enumerate_pixels.return_value = all_pixels
    mock_patch.to_movie_coordinate = lambda x: (2,) if x == (0,) else (3,)
    mock_embedding.get_vector = lambda x: x

    mock_edge_selector.select_edges.return_value = [((0,), (1,))]

    GC = GraphConstructor(mock_edge_selector, lambda x, y: 1)
    graph = GC.construct(mock_patch, mock_embedding)

    assert set(graph.nodes) == all_pixels
    assert graph[(2,)][(3,)]["weight"] == 1
    assert graph[(3,)][(2,)]["weight"] == 1


def test_sparse_computation_select_edges(mock_embedding):
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
