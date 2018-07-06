import pytest
import os
import numpy as np

TEST_DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "test_data"
)


@pytest.fixture
def M():
    """Simple movie for testing.

    Movie consists of three images of 5 x 10 pixels. All pixels in the first
    image have value 1, all pixels in the second image have value 2, and all
    pixels in the third image have value 3.
    """
    from hnccorr.movie import Movie

    return Movie(
        "Simple",
        image_dir=str(os.path.join(TEST_DATA_DIR, "simple_movie")),
        num_images=3,
    )


def test_name(M):
    assert M.name == "Simple"


def test_num_frames(M):
    assert M.num_frames == 3


def test_num_pixels(M):
    assert M.num_pixels == 50


def test_data_size(M):
    assert M.data_size == (3, 5, 10)


def test_pixel_size(M):
    assert M.pixel_size == (5, 10)


def test_is_valid_pixel_index(M):
    assert not M.is_valid_pixel_index((0, -1))
    assert not M.is_valid_pixel_index((4, 11))
    assert M.is_valid_pixel_index((4, 9))


def test_movie_get_item(M):
    assert M[0, 0, 0] == 1.0
    np.testing.assert_allclose(M[2, :, :], np.ones((5, 10)) * 3)
