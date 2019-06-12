import pytest

from hnccorr.seeds import PositiveSeedSelector, NegativeSeedSelector


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
def test_positive_seed_selector(center_seed, radius, expected_seeds):
    assert PositiveSeedSelector(radius).select(center_seed, (5, 5)) == expected_seeds


@pytest.mark.parametrize(
    "radius, count, center_seed, movie_size, expected_seeds",
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
def test_select_negative_seeds(radius, count, center_seed, movie_size, expected_seeds):
    assert (
        NegativeSeedSelector(radius, count).select(center_seed, movie_size)
        == expected_seeds
    )


def test_select_negative_seeds_invalid_dimension():
    with pytest.raises(ValueError):
        NegativeSeedSelector(2, 4).select((5, 5, 5), (10, 10, 10))

    with pytest.raises(ValueError):
        NegativeSeedSelector(2, 4).select((5,), (10,))
