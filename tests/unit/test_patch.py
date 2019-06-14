import pytest
import numpy as np
from copy import copy, deepcopy
from pytest_mock import mocker

from hnccorr.patch import Patch


@pytest.fixture
def simple_patch(MM):
    return Patch(MM, (9,), 7)


def test_even_windowsize(MM):
    with pytest.raises(ValueError):
        Patch(MM, (5,), 6)


def test_pixel_shape(simple_patch):
    assert simple_patch.pixel_shape == (7,)


def test_num_frames(simple_patch):
    assert simple_patch.num_frames == 3


def test_data(simple_patch, MM):
    np.testing.assert_equal(simple_patch[:], MM[:, 3:10])


def test_patch_enumerate_pixels(simple_patch, MM2):
    assert simple_patch.enumerate_pixels() == {(3,), (4,), (5,), (6,), (7,), (8,), (9,)}
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


@pytest.mark.parametrize(
    "movie_coordinate, patch_coordinate", [((6,), (3,)), ((3,), (0,))]
)
def test_to_movie_coordinate(movie_coordinate, patch_coordinate, simple_patch, MM):
    assert simple_patch.to_movie_coordinate(patch_coordinate) == movie_coordinate


@pytest.mark.parametrize(
    "movie_coordinate, patch_coordinate", [((9,), (6,)), ((3,), (0,))]
)
def test_to_patch_coordinate(movie_coordinate, patch_coordinate, simple_patch, MM):
    assert simple_patch.to_patch_coordinate(movie_coordinate) == patch_coordinate
