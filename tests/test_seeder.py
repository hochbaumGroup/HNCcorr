import pytest


@pytest.fixture
def LCS(MM):
    from hnccorr.seeder import LocalCorrelationSeeder

    return LocalCorrelationSeeder(MM, neighborhood_size=3, keep_fraction=0.2)


def test_local_corr_seeder(LCS):
    assert LCS.next() == (9,)
    assert LCS.next() == (8,)
    assert LCS.next() is None
