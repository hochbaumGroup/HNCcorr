import pytest
from hnccorr.graph import GraphConstructor


@pytest.fixture
def MES():
    class MockEdgeSelector:
        def select_edges(self, embedding):
            return [((0,), (1,)), ((0,), (2,))]

    return MockEdgeSelector()


@pytest.fixture
def MW():
    return lambda x, a, b: b[0]


def test_graph_constructor(P, MES, MW):
    GC = GraphConstructor(MES, MW)
    G = GC.construct(P((0,)), None)

    num_nodes = 7
    assert len(G.nodes) == num_nodes
    for i in range(num_nodes):
        assert (i,) in G.nodes

    assert len(G.edges) == 2
    assert ((0,), (1,)) in G.edges
    assert ((0,), (2,)) in G.edges
    assert G[(0,)][(1,)]["weight"] == 1
    assert G[(0,)][(2,)]["weight"] == 2


def test_graph_constructor_nodes_offset_from_zero(mocker, dummy):
    all_pixels = {(2,), (3,), (4,), (5,), (6,), (7,), (8,)}
    Patch = mocker.patch("hnccorr.patch.Patch", autospec=True)
    Patch.return_value.enumerate_pixels.return_value = all_pixels

    EdgeSelector = mocker.patch(
        "hnccorr.edge_selection.SparseComputation", autospec=True
    )
    EdgeSelector.return_value.select_edges.return_value = [((0,), (1,))]

    GC = GraphConstructor(EdgeSelector(dummy, dummy), lambda emb, x, y: 1)
    mock_patch = Patch(dummy, dummy, dummy)
    graph = GC.construct(mock_patch, None)

    assert set(graph.nodes) == all_pixels
    assert graph[(2,)][(3,)]["weight"] == 1
    assert graph[(3,)][(2,)]["weight"] == 1
