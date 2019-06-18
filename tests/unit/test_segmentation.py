import pytest
from copy import copy
from itertools import product

from hnccorr.segmentation import Segmentation


def test_weight():
    assert Segmentation({0, 1}, 0.5).weight == 0.5


def test_selection():
    assert Segmentation({(0, 1)}, 0.5).selection == {(0, 1)}


def test_equality_wrong_class():
    class FakeSegmentation:
        pass

    assert Segmentation({(0, 1)}, 0.5) != FakeSegmentation()


def test_clean_fill_hole():
    selection = {(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (1, 2), (2, 2)}
    weight = 1
    original = Segmentation(selection, weight)

    new = original.clean(set(), (5, 5))
    assert new == Segmentation(
        {(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)}, weight
    )
    assert original == Segmentation(selection, weight)


def test_clean_select_seed_component():
    selection = {(0, 0), (3, 3)}
    weight = 1
    original = Segmentation(selection, weight)

    new = original.clean({(3, 3)}, (5, 5))
    assert new == Segmentation({(3, 3)}, weight)
    assert original == Segmentation(selection, weight)
