import pytest


@pytest.fixture
def MES():
    class MockEdgeSelector:
        def select_edges():
            return [((0,), (1,)), ((0,), (2,))]

    return MockEdgeSelector()


@pytest.fixture
def ME():
    class MockEmbedding:
        def distance(self, first, second):
            return second[0]


def test_graph_constructor(P, MES, ME):
    from hnccorr.graph import GraphConstructor

    GC = GraphConstructor(P((0,)), MES, ME)
    G = GC.construct()

    num_nodes = 7
    assert len(G.nodes) == num_nodes
    for i in range(num_nodes):
        assert (i,) in G.nodes

    assert len(G.edges) == 2
    assert ((0,), (1,)) in G.edges
    assert ((0,), (2,)) in G.edges
    assert G[(0,)][(1,)]["weight"] == 1
    assert G[(0,)][(1,)]["weight"] == 2
