import pytest
from copy import copy

from hnccorr.segmentation import Segmentation


def test_weight():
    assert Segmentation({0, 1}, 0.5).weight == 0.5


def test_selection():
    assert Segmentation({(0, 1)}, 0.5).selection == {(0, 1)}


def test_fill_hole():
    assert Segmentation(
        {(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (1, 2), (2, 2)}, 0.5
    ).clean(set(), (5, 5)).selection == {
        (0, 0),
        (1, 0),
        (2, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (0, 2),
        (1, 2),
        (2, 2),
    }


def test_clean_select_seed_component():
    assert Segmentation({(0, 0), (3, 3)}, 0.5).clean(
        {(3, 3)}, (5, 5)
    ).selection == {(3, 3)}
