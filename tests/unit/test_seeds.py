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

from hnccorr.seeds import (
    PositiveSeedSelector,
    NegativeSeedSelector,
    LocalCorrelationSeeder,
)


@pytest.fixture
def LCS():
    return LocalCorrelationSeeder(3, 0.2, 2, 1)


def extract_valid_pixels_10_10(pixels):
    return pixels.intersection({(i, j) for i in range(10) for j in range(10)})


class TestLocalCorrelationSeeder:
    def test_local_corr_seeder(self, LCS, MM):
        LCS.select_seeds(MM)
        assert LCS.next() == (9,)
        assert LCS.next() == (8,)
        assert LCS.next() is None

    def test_local_corr_seeder_grid_size(self, MM):
        lcs = LocalCorrelationSeeder(3, 1.0, 2, 5)
        lcs.select_seeds(MM)
        assert lcs.next() == (9,)
        assert lcs.next() == (4,)
        assert lcs.next() is None

    def test_local_corr_seeder_width_not_divisable_by_grid_size(self, MM):
        lcs = LocalCorrelationSeeder(3, 1.0, 2, 4)
        lcs.select_seeds(MM)
        assert lcs.next() == (9,)
        assert lcs.next() == (7,)
        assert lcs.next() == (3,)
        assert lcs.next() is None

    def test_local_corr_seeder_reset(self, LCS, MM):
        LCS.select_seeds(MM)
        assert LCS.next() == (9,)

        LCS.reset()
        assert LCS.next() == (9,)

    def test_seeder_exclude_pixels(self, LCS, MM):
        LCS.select_seeds(MM)
        assert LCS.next() == (9,)
        LCS.exclude_pixels({(8,)})
        assert LCS.next() is None

    def test_seeder_exclude_pixels_boundary(self, LCS, MM):
        LCS.select_seeds(MM)
        assert LCS.next() == (9,)
        LCS.exclude_pixels({(6,)})
        assert LCS.next() is None

    def test_seeder_reset_excluded_pixels(self, LCS, MM):
        LCS.select_seeds(MM)
        assert LCS.next() == (9,)
        LCS.exclude_pixels({(9,)})
        LCS.reset()
        assert LCS.next() == (9,)

    def test_seeder_select_seeds_should_reset_excluded_pixels(self, LCS, MM):
        LCS.select_seeds(MM)
        LCS.exclude_pixels({(9,)})
        LCS.select_seeds(MM)
        assert LCS.next() == (9,)


class TestPositiveSeedSelector:
    @pytest.mark.parametrize(
        "center_seed, radius, expected_seeds",
        [
            ((0, 0), 0, {(0, 0)}),
            ((0, 0), 1, {(0, 0), (0, 1), (1, 1), (1, 0)}),
            (
                (1, 1),
                1,
                {
                    (0, 0),
                    (0, 1),
                    (0, 2),
                    (1, 0),
                    (1, 1),
                    (1, 2),
                    (2, 0),
                    (2, 1),
                    (2, 2),
                },
            ),
        ],
    )
    def test_positive_seed_selector(
        self, center_seed, radius, expected_seeds, mock_movie
    ):
        mock_movie.num_dimensions = 2
        mock_movie.pixel_size = (10, 10)
        mock_movie.extract_valid_pixels = extract_valid_pixels_10_10

        assert (
            PositiveSeedSelector(radius).select(center_seed, mock_movie)
            == expected_seeds
        )


class TestNegativeSeedSelector:
    @pytest.mark.parametrize(
        "radius, count, center_seed, movie_pixel_size, expected_seeds",
        [
            (0, 8, (5, 5), (10, 10), {(5, 5)}),
            (2, 4, (0, 0), (10, 10), {(0, 2), (2, 0)}),
            (2, 4, (9, 9), (10, 10), {(9, 7), (7, 9)}),
            (
                2,
                8,
                (5, 5),
                (10, 10),
                {(3, 5), (4, 4), (7, 5), (4, 6), (5, 3), (6, 4), (5, 7), (6, 6)},
            ),
            (2, 4, (5, 5), (10, 10), {(3, 5), (7, 5), (5, 3), (5, 7)}),
        ],
    )
    def test_select_negative_seeds(
        self, radius, count, center_seed, movie_pixel_size, expected_seeds, mock_movie
    ):
        mock_movie.num_dimensions = len(movie_pixel_size)
        mock_movie.pixel_size = movie_pixel_size
        mock_movie.extract_valid_pixels = extract_valid_pixels_10_10

        assert (
            NegativeSeedSelector(radius, count).select(center_seed, mock_movie)
            == expected_seeds
        )

    @pytest.mark.parametrize(
        "radius, count, center_seed, movie_pixel_size",
        [(2, 4, (5, 5, 5), (10, 10, 10)), (2, 4, (5,), (10,))],
    )
    def test_select_negative_seeds_invalid_dimension(
        self, radius, count, center_seed, movie_pixel_size, mock_movie
    ):
        mock_movie.num_dimensions = len(movie_pixel_size)
        mock_movie.pixel_size = movie_pixel_size

        with pytest.raises(ValueError):
            NegativeSeedSelector(radius, count).select(center_seed, mock_movie)
