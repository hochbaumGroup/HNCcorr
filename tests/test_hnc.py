import networkx as nx

from hnccorr.utils import generate_pixels


def test_hnc(P):
    from hnccorr.hnc import HNC
    from hnccorr.segmentation import Segmentation

    patch = P((0,))

    G = nx.Graph()
    G.add_nodes_from(generate_pixels(patch.pixel_size))

    h = HNC(patch, G, "weight")

    segms = h.solve_parametric(0, 2)
    seg = segms[0]
    assert len(segms) == 1
    assert seg.weight == 2.
    # all except negative seeds
    assert seg.selection == {(0,), (1,), (3,), (4,), (5,), (6,)}
    assert seg._patch == patch
