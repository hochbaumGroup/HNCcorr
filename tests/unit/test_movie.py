import pytest
import os
import numpy as np

from hnccorr.movie import Movie

from conftest import TEST_DATA_DIR


@pytest.fixture
def M():
    data = np.zeros((3, 5, 10), np.uint16)
    data[0, :, :] = np.ones((5, 10))
    data[1, :, :] = np.ones((5, 10)) * 2
    data[2, :, :] = np.ones((5, 10)) * 3

    return Movie("Simple", data)


def test_movie_from_tiff_images(M):
    """
    Movie consists of three images of 5 x 10 pixels. All pixels in the first
    image have value 1, all pixels in the second image have value 2, and all
    pixels in the third image have value 3.
    """
    movie_from_tiff = Movie.from_tiff_images(
        "Simple",
        image_dir=str(os.path.join(TEST_DATA_DIR, "simple_movie")),
        num_images=3,
    )

    # compare data of movie from_tiff and direct initialization.
    np.testing.assert_allclose(movie_from_tiff[:], M[:])


def test_name(M):
    assert M.name == "Simple"


def test_num_frames(M):
    assert M.num_frames == 3


def test_num_pixels(M):
    assert M.num_pixels == 50


def test_data_size(M):
    assert M.data_size == (3, 5, 10)


def test_pixel_shape(M):
    assert M.pixel_shape == (5, 10)


def test_is_valid_pixel_coordinate(M):
    assert not M.is_valid_pixel_coordinate((0, -1))
    assert not M.is_valid_pixel_coordinate((4, 11))
    assert not M.is_valid_pixel_coordinate((4,))
    assert M.is_valid_pixel_coordinate((4, 9))


def test_extract_valid_pixels(M):
    assert M.extract_valid_pixels({(0, 0), (-1, 0), (4, 10)}) == {(0, 0)}


def test_movie_get_item(M):
    assert M[0, 0, 0] == 1.0
    np.testing.assert_allclose(M[2, :, :], np.ones((5, 10)) * 3)
