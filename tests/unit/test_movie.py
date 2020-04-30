# Copyright © 2017. Regents of the University of California (Regents). All Rights
# Reserved.
#
# Permission to use, copy, modify, and distribute this software and its documentation
# for educational, research, and not-for-profit purposes, without fee and without a
# signed licensing agreement, is hereby granted, provided that the above copyright
# notice, this paragraph and the following two paragraphs appear in all copies,
# modifications, and distributions. Contact The Office of Technology Licensing, UC
# Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201,
# for commercial licensing opportunities. Created by Quico Spaen, Roberto Asín-Achá,
# and Dorit S. Hochbaum, Department of Industrial Engineering and Operations Research,
# University of California, Berkeley.
#
# IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
# INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE
# OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE
# SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS
# IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES,
# ENHANCEMENTS, OR MODIFICATIONS.
import pytest
import os
import numpy as np
from copy import copy, deepcopy
from pytest_mock import mocker

from hnccorr.movie import Movie, Patch, Subsampler

from conftest import TEST_DATA_DIR


@pytest.fixture
def movie_data():
    data = np.zeros((3, 5, 10), np.float32)
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


@pytest.fixture
def subsampler():
    return Subsampler((800, 10, 10), 10)


class TestMovie:
    def test_movie_from_tiff_images(self, movie_data):
        """
        Movie consists of three images of 5 x 10 pixels. All pixels in the first
        image have value 1, all pixels in the second image have value 2, and all
        pixels in the third image have value 3.
        """
        movie_from_tiff = Movie.from_tiff_images(
            "Simple",
            image_dir=str(os.path.join(TEST_DATA_DIR, "simple_movie")),
            num_images=3,
            subsample=1,
        )

        # compare data of movie from_tiff and direct initialization.
        np.testing.assert_allclose(movie_from_tiff[:], movie_data)

    def test_movie_from_tiff_images_with_subsampling(self, movie_data):
        """
        Movie consists of three images of 5 x 10 pixels. All pixels in the first
        image have value 1, all pixels in the second image have value 2, and all
        pixels in the third image have value 3.
        """
        movie_from_tiff = Movie.from_tiff_images(
            "Simple",
            image_dir=str(os.path.join(TEST_DATA_DIR, "simple_movie_long")),
            num_images=21,
            subsample=2,
        )

        # compare data of movie from_tiff and direct initialization.

        data = np.zeros((11, 5, 10), np.float32)
        data[0, :, :] = np.ones((5, 10)) * 1.5
        for i in range(1, 11):
            data[i, :, :] = np.ones((5, 10)) * 2
        np.testing.assert_allclose(movie_from_tiff[:], data)

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
            subsample=1,
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

    def test_data_access_float64(self, M):
        assert M[:].dtype == np.float64

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


class TestSubsampler:
    def test_output_shape(self):
        assert Subsampler((800, 512, 512), 10).output_shape == (80, 512, 512)
        assert Subsampler((589, 128, 256), 10).output_shape == (59, 128, 256)
        assert Subsampler((589, 128, 256), 25).output_shape == (24, 128, 256)

    def test_buffer(self, subsampler):
        np.testing.assert_allclose(subsampler.buffer, np.zeros((10, 10, 10)))

        np.testing.assert_allclose(
            Subsampler((80, 512, 512), 10).buffer, np.zeros((8, 512, 512))
        )

    def test_buffer_last_frames(self, subsampler):
        subsampler = Subsampler((21, 10, 10), 2)
        subsampler.advance_buffer()

        assert subsampler.buffer.shape[0] == 1

    def test_add_frame(self, subsampler):
        subsampler.add_frame(np.ones((10, 10)))

        np.testing.assert_allclose(subsampler.buffer[0, :, :], np.ones((10, 10)))
        np.testing.assert_allclose(subsampler.buffer[1:, :, :], np.zeros((9, 10, 10)))

        subsampler.add_frame(np.ones((10, 10)) * 2)

        np.testing.assert_allclose(subsampler.buffer[0, :, :], 1.5 * np.ones((10, 10)))
        np.testing.assert_allclose(subsampler.buffer[1:, :, :], np.zeros((9, 10, 10)))

    def test_advance_to_next_frame_(self, subsampler):
        for i in range(10):
            subsampler.add_frame(np.ones((10, 10)))

        np.testing.assert_allclose(subsampler.buffer[0, :, :], np.ones((10, 10)))
        np.testing.assert_allclose(subsampler.buffer[1:, :, :], np.zeros((9, 10, 10)))

        subsampler.add_frame(np.ones((10, 10)) * 2)

        np.testing.assert_allclose(subsampler.buffer[0, :, :], np.ones((10, 10)))
        np.testing.assert_allclose(subsampler.buffer[1, :, :], 2 * np.ones((10, 10)))
        np.testing.assert_allclose(subsampler.buffer[2:, :, :], np.zeros((8, 10, 10)))

    def test_buffer_full(self, subsampler):
        for i in range(100):
            assert subsampler.buffer_full == False
            subsampler.add_frame(np.ones((10, 10)))
        assert subsampler.buffer_full == True

    def test_buffer_indices(self, subsampler):
        assert subsampler.buffer_indices == (0, 10)

    def test_advance_buffer(self, subsampler):
        for i in range(100):
            subsampler.add_frame(np.ones((10, 10)))

        subsampler.advance_buffer()
        np.testing.assert_allclose(subsampler.buffer, np.zeros((10, 10, 10)))
        assert subsampler.buffer_full == False
        assert subsampler.buffer_indices == (10, 20)

    def test_advance_buffer_resets_frame_count(self, subsampler):
        for i in range(100):
            subsampler.add_frame(np.ones((10, 10)))

        subsampler.advance_buffer()

        subsampler.add_frame(np.ones((10, 10)))

        np.testing.assert_allclose(subsampler.buffer[0, :, :], np.ones((10, 10)))

    def test_buffer_indices_last_frames(self):
        subsampler = Subsampler((21, 10, 10), 2)

        subsampler.advance_buffer()
        assert subsampler.buffer_indices == (10, 11)

    def test_prevent_buffer_overflow(self, subsampler):
        for i in range(100):
            subsampler.add_frame(np.ones((10, 10)))

        with pytest.raises(ValueError):
            subsampler.add_frame(np.ones((10, 10)))
