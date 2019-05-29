import pytest
import networkx as nx

from hnccorr.utils import generate_pixels
from hnccorr.hnc import HncParametric
from hnccorr.seeds import Seeds


def test_hnc():
    G = nx.Graph()
    G.add_nodes_from(generate_pixels((7,)))

    h = HncParametric(0, 2)

    segmentations = h.solve(Seeds((3,), {(2,), (3,), (4,)}, {(1,), (5,)}), G)

    assert len(segmentations) == 1
    assert segmentations[0].selection == {(0,), (2,), (3,), (4,), (6,)}
    assert segmentations[0].weight == pytest.approx(2.0)
