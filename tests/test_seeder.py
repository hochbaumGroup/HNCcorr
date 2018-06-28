import pytest


@pytest.fixture
def LCS(MM):
    from hnccorr.seeder import LocalCorrelationSeeder

    return LocalCorrelationSeeder(MM, 3, 3, 0.2)


def test_local_corr_seeder(LCS):
    patch = LCS.next()
    assert patch.positive_seeds == set([(0,), (1,)])

    patch = LCS.next()
    assert patch.positive_seeds == set([(0,), (1,), (2,)])

    assert LCS.next() is None
