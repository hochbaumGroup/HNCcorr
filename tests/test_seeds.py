import pytest

from hnccorr.seeds import Seeds
from hnccorr.seeds import PositiveSeedSelector, NegativeSeedSelector


@pytest.fixture
def neg_seeds():
    return {(5,), (6,)}


@pytest.fixture
def S(pos_seeds, neg_seeds):
    return Seeds((5,), pos_seeds, neg_seeds)


@pytest.fixture
def empty_seed():
    return Seeds(None, [], [])


def test_center_seed(S):
    assert S.center_seed == (5,)


def test_positive_seeds(S, pos_seeds):
    assert S.positive_seeds == pos_seeds


def test_negative_seeds(S, neg_seeds):
    assert S.negative_seeds == neg_seeds


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
def test_select_positive_seeds(empty_seed, center_seed, radius, expected_seeds):
    empty_seed.center_seed = center_seed
    assert empty_seed.select_positive_seeds(radius, (5, 5)) == expected_seeds


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
    assert PositiveSeedSelector(radius, (5, 5)).select(center_seed) == expected_seeds


def test_select_negative_seeds_simple(empty_seed):
    empty_seed.center_seed = (5, 5)
    assert empty_seed.select_negative_seeds(2, 4, (10, 10)) == {
        (3, 5),
        (7, 5),
        (5, 3),
        (5, 7),
    }


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
