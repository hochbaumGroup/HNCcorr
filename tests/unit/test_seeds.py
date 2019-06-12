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


def test_select_negative_seed_selector():
    assert NegativeSeedSelector(2, 4, (10, 10)).select((5, 5)) == {
        (3, 5),
        (7, 5),
        (5, 3),
        (5, 7),
    }


def test_select_negative_seeds_zero_radius():
    assert NegativeSeedSelector(0, 8, (10, 10)).select((5, 5)) == {(5, 5)}


def test_select_negative_seeds_count_rounding():
    assert NegativeSeedSelector(2, 8, (10, 10)).select((5, 5)) == {
        (3, 5),
        (4, 4),
        (7, 5),
        (4, 6),
        (5, 3),
        (6, 4),
        (5, 7),
        (6, 6),
    }


def test_select_negative_seeds_topleft_corner():
    assert NegativeSeedSelector(2, 4, (10, 10)).select((0, 0)) == {(0, 2), (2, 0)}


def test_select_negative_seeds_bottomright_corner():
    assert NegativeSeedSelector(2, 4, (10, 10)).select((9, 9)) == {(7, 9), (9, 7)}


def test_select_negative_seeds_invalid_dimension():
    with pytest.raises(ValueError):
        NegativeSeedSelector(2, 4, (10, 10, 10))

    with pytest.raises(ValueError):
        NegativeSeedSelector(2, 4, (10,))
