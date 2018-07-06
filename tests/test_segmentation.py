import pytest


def test_weight(S, SS1):
    assert S(SS1).weight == 0.5


def test_selection(S, SS1):
    assert S(SS1).selection == SS1


def test_clean(S, SS1):
    segmentation = S(SS1)
    segmentation.clean()
    assert segmentation.selection == {
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
