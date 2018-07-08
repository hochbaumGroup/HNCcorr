import networkx as nx

from hnccorr.utils import generate_pixels


def test_hnc(P1, S4):
    from hnccorr.hnc import HNC
    from hnccorr.segmentation import Segmentation

    G = nx.Graph()
    G.add_nodes_from(generate_pixels(P1.pixel_size))

    h = HNC(P1, G, "weight")

    assert h.solve_parametric(0, 2) == [S4]
