import pytest
from copy import copy


def test_weight(S1):
    assert S1.weight == 0.5


def test_selection(S1, SS1):
    assert S1.selection == SS1


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


def test_equal(S1, S2):
    assert S1 == copy(S1)
    assert S1 != S2
