from copy import copy
from itertools import product
import pytest
import networkx as nx


from hnccorr.segmentation import HncParametricWrapper, Segmentation


def test_hnc():
    G = nx.Graph()
    G.add_nodes_from((i,) for i in range(7))

    h = HncParametricWrapper(0, 2)

    segmentations = h.solve(G, {(2,), (3,), (4,)}, {(1,), (5,)})

    assert len(segmentations) == 1
    assert segmentations[0].selection == {(0,), (2,), (3,), (4,), (6,)}
    assert segmentations[0].weight == pytest.approx(2.0)


def test_segmentation_weight():
    assert Segmentation({0, 1}, 0.5).weight == 0.5


def test_segmentation_selection():
    assert Segmentation({(0, 1)}, 0.5).selection == {(0, 1)}


def test_segmentation_equality_wrong_class():
    class FakeSegmentation:
        pass

    assert Segmentation({(0, 1)}, 0.5) != FakeSegmentation()


def test_segmentation_clean_fill_hole():
    selection = {(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (1, 2), (2, 2)}
    weight = 1
    original = Segmentation(selection, weight)

    new = original.clean(set(), (5, 5))
    assert new == Segmentation(
        {(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)}, weight
    )
    assert original == Segmentation(selection, weight)


def test_segmentation_clean_select_seed_component():
    selection = {(0, 0), (3, 3)}
    weight = 1
    original = Segmentation(selection, weight)

    new = original.clean({(3, 3)}, (5, 5))
    assert new == Segmentation({(3, 3)}, weight)
    assert original == Segmentation(selection, weight)
