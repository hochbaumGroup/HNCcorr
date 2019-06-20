import pytest

from hnccorr.seeds import (
    PositiveSeedSelector,
    NegativeSeedSelector,
    LocalCorrelationSeeder,
)


@pytest.fixture
def mock_movie(mocker, dummy):
    movie = mocker.patch("hnccorr.movie.Movie", autospec=True)(dummy, dummy)
    return movie


@pytest.fixture
def LCS():
    return LocalCorrelationSeeder(3, 0.2, 2)


def extract_valid_pixels_10_10(pixels):
    return pixels.intersection({(i, j) for i in range(10) for j in range(10)})


class TestLocalCorrelationSeeder:
    def test_local_corr_seeder(self, LCS, MM):
        LCS.select_seeds(MM)
        assert LCS.next() == (9,)
        assert LCS.next() == (8,)
        assert LCS.next() is None

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
