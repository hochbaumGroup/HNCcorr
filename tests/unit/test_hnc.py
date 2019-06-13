import pytest
import networkx as nx

from hnccorr.hnc import HncParametric


def test_hnc():
    G = nx.Graph()
    G.add_nodes_from((i,) for i in range(7))

    h = HncParametric(0, 2)

    segmentations = h.solve(G, {(2,), (3,), (4,)}, {(1,), (5,)})

    assert len(segmentations) == 1
    assert segmentations[0].selection == {(0,), (2,), (3,), (4,), (6,)}
    assert segmentations[0].weight == pytest.approx(2.0)
