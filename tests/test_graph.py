import pytest


@pytest.fixture
def MES():
    class MockEdgeSelector:
        def select_edges(self, embedding):
            return [((0,), (1,)), ((0,), (2,))]

    return MockEdgeSelector()


@pytest.fixture
def MW():
    return lambda a, b: b[0]


def test_graph_constructor(P, MES, MW):
    from hnccorr.graph import GraphConstructor

    GC = GraphConstructor(P((0,)), MES, MW)
    G = GC.construct(None)

    num_nodes = 7
    assert len(G.nodes) == num_nodes
    for i in range(num_nodes):
        assert (i,) in G.nodes

    assert len(G.edges) == 2
    assert ((0,), (1,)) in G.edges
    assert ((0,), (2,)) in G.edges
    assert G[(0,)][(1,)]["weight"] == 1
    assert G[(0,)][(2,)]["weight"] == 2
