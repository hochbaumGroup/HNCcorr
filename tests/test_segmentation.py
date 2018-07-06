import pytest


def test_weight(S1):
    assert S1.weight == 0.5


def test_selection(S1):
    assert S1.selection == {
        (0, 0),
        (1, 0),
        (2, 0),
        (0, 1),
        (2, 1),
        (0, 2),
        (1, 2),
        (2, 2),
    }


def test_clean(S1):
    S1.clean()
    assert S1.selection == {
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
