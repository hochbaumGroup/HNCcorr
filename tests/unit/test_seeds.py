import pytest

from hnccorr.seeds import PositiveSeedSelector, NegativeSeedSelector


@pytest.fixture
def mock_movie(mocker, dummy):
    movie = mocker.patch("hnccorr.movie.Movie", autospec=True)(dummy, dummy)
    return movie


def extract_valid_pixels_10_10(pixels):
    return pixels.intersection({(i, j) for i in range(10) for j in range(10)})


@pytest.mark.parametrize(
    "center_seed, radius, expected_seeds",
    [
        ((0, 0), 0, {(0, 0)}),
        ((0, 0), 1, {(0, 0), (0, 1), (1, 1), (1, 0)}),
        (
            (1, 1),
            1,
            {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)},
        ),
    ],
)
def test_positive_seed_selector(center_seed, radius, expected_seeds, mock_movie):
    mock_movie.num_dimensions = 2
    mock_movie.pixel_size = (10, 10)
    mock_movie.extract_valid_pixels = extract_valid_pixels_10_10

    assert (
        PositiveSeedSelector(radius).select(center_seed, mock_movie) == expected_seeds
    )


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
    radius, count, center_seed, movie_pixel_size, expected_seeds, mock_movie
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
    radius, count, center_seed, movie_pixel_size, mock_movie
):
    mock_movie.num_dimensions = len(movie_pixel_size)
    mock_movie.pixel_size = movie_pixel_size

    with pytest.raises(ValueError):
        NegativeSeedSelector(radius, count).select(center_seed, mock_movie)
