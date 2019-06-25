import pytest
import os
import numpy as np
from copy import copy, deepcopy
from pytest_mock import mocker

from hnccorr.movie import Movie, Patch

from conftest import TEST_DATA_DIR


@pytest.fixture
def movie_data():
    data = np.zeros((3, 5, 10), np.uint16)
    data[0, :, :] = np.ones((5, 10))
    data[1, :, :] = np.ones((5, 10)) * 2
    data[2, :, :] = np.ones((5, 10)) * 3
    return data


@pytest.fixture
def M(movie_data):
    return Movie("Simple", movie_data)


@pytest.fixture
def simple_patch(MM):
    return Patch(MM, (9,), 7)


class TestMovie:
    def test_movie_from_tiff_images(self, M):
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

    def test_movie_from_tiff_images_memmap(self, movie_data):
        """
        Movie consists of three images of 5 x 10 pixels. All pixels in the first
        image have value 1, all pixels in the second image have value 2, and all
        pixels in the third image have value 3.
        """
        movie_from_tiff = Movie.from_tiff_images(
            "Simple",
            image_dir=str(os.path.join(TEST_DATA_DIR, "simple_movie")),
            num_images=3,
            memmap=True,
        )

        # compare data of movie from_tiff and direct initialization.
        np.testing.assert_allclose(movie_from_tiff[:], movie_data)

    def test_movie_name(self, M):
        assert M.name == "Simple"

    def test_movie_num_frames(self, M):
        assert M.num_frames == 3

    def test_movie_num_pixels(self, M):
        assert M.num_pixels == 50

    def test_movie_data_size(self, M):
        assert M.data_size == (3, 5, 10)

    def test_movie_pixel_shape(self, M):
        assert M.pixel_shape == (5, 10)

    def test_movie_is_valid_pixel_coordinate(self, M):
        assert not M.is_valid_pixel_coordinate((0, -1))
        assert not M.is_valid_pixel_coordinate((4, 11))
        assert not M.is_valid_pixel_coordinate((4,))
        assert M.is_valid_pixel_coordinate((4, 9))

    def test_movie_extract_valid_pixels(self, M):
        assert M.extract_valid_pixels({(0, 0), (-1, 0), (4, 10)}) == {(0, 0)}

    def test_movie_get_item(self, M, movie_data):
        assert M[0, 0, 0] == 1.0
        np.testing.assert_allclose(M[2, :, :], movie_data[2, :, :])

    def test_movie_init_with_memmap(self, movie_data):
        # prepare memmapped file
        filename = os.path.join(TEST_DATA_DIR, "test_memdata.npy")
        mem_data = np.memmap(filename, dtype=np.uint16, mode="w+", shape=(3, 5, 10))
        mem_data[:] = movie_data[:]

        del mem_data

        # load memmapped file
        mem_data = np.memmap(filename, dtype=np.uint16, mode="r", shape=(3, 5, 10))
        movie = Movie("Memmap", mem_data)

        np.testing.assert_allclose(movie[:], movie_data)


class TestPatch:
    def test_patch_pixel_shape(self, simple_patch):
        assert simple_patch.pixel_shape == (7,)

    def test_patch_num_frames(self, simple_patch):
        assert simple_patch.num_frames == 3

    def test_patch_data(self, simple_patch, MM):
        np.testing.assert_equal(simple_patch[:], MM[:, 3:10])

    def test_patch_even_windowsize(self, MM):
        with pytest.raises(ValueError):
            Patch(MM, (5,), 6)

    def test_patch_enumerate_pixels(self, simple_patch, MM2):
        assert simple_patch.enumerate_pixels() == {
            (3,),
            (4,),
            (5,),
            (6,),
            (7,),
            (8,),
            (9,),
        }
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
    def test_patch_to_movie_coordinate(
        self, movie_coordinate, patch_coordinate, simple_patch, MM
    ):
        assert simple_patch.to_movie_coordinate(patch_coordinate) == movie_coordinate

    @pytest.mark.parametrize(
        "movie_coordinate, patch_coordinate", [((9,), (6,)), ((3,), (0,))]
    )
    def test_patch_to_patch_coordinate(
        self, movie_coordinate, patch_coordinate, simple_patch, MM
    ):
        assert simple_patch.to_patch_coordinate(movie_coordinate) == patch_coordinate
