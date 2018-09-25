import pytest


@pytest.fixture
def S(pos_seeds):
    from hnccorr.seeds import Seeds

    return Seeds(pos_seeds)


def test_positive_seeds(S, pos_seeds):
    assert S.positive_seeds == pos_seeds
