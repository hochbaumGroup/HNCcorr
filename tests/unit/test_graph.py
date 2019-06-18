import pytest
from hnccorr.graph import GraphConstructor


@pytest.fixture
def mock_patch(mocker, dummy):
    return mocker.patch("hnccorr.patch.Patch", autospec=True)(dummy, dummy, dummy)


@pytest.fixture
def mock_edge_selector(mocker, dummy):
    return mocker.patch(
        "hnccorr.edge_selection.SparseComputationEmbeddingWrapper", autospec=True
    )(dummy, dummy)


@pytest.fixture
def mock_embedding(mocker, dummy):
    return mocker.patch("hnccorr.embedding.CorrelationEmbedding", autospec=True)(dummy)


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
