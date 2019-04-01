import pytest
import networkx as nx

from hnccorr.utils import generate_pixels
from hnccorr.hnc import HNC
from hnccorr.seeds import Seeds


def test_hnc(P1):
    G = nx.Graph()
    G.add_nodes_from(generate_pixels(P1.pixel_size))

    h = HNC(Seeds((3,), {(2,), (3,), (4,)}, {(1,), (5,)}), G, "weight")

    segmentations = h.solve_parametric(0, 2)

    assert len(segmentations) == 1
    assert segmentations[0].selection == {(0,), (2,), (3,), (4,), (6,)}
    assert segmentations[0].weight == pytest.approx(2.0)
