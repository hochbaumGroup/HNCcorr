import pytest
import numpy as np
from copy import copy, deepcopy
from pytest_mock import mocker

from hnccorr.patch import Patch


def test_even_windowsize(MM, pos_seeds):
    with pytest.raises(ValueError):
        Patch(MM, (5,), 6)


def test_pixel_size(P1):
    assert P1.pixel_size == (7,)


def test_num_frames(P1):
    assert P1.num_frames == 3


def test_data(P1, MM):
    np.testing.assert_equal(P1[:], MM[:, 2:9])


def test_patch_enumerate_pixels(MM, MM2):
    assert Patch(MM, (5,), 3).enumerate_pixels() == {(4,), (5,), (6,)}
    assert Patch(MM2, (5, 5), 3).enumerate_pixels() == {
        (4, 4),
        (4, 5),
        (4, 6),
        (5, 4),
        (5, 5),
        (5, 6),
        (6, 4),
        (6, 5),
        (6, 6),
    }


def test_to_movie_index(P):
    patch = P((9,))
    patch.to_movie_index((3,)) == (6,)
    patch.to_movie_index((0,)) == (3,)


def test_to_patch_index(P):
    patch = P((9,))
    patch.to_patch_index((9,)) == (6,)
    patch.to_patch_index((3,)) == (0,)
