import pytest

from hnccorr.seeder import LocalCorrelationSeeder


def test_local_corr_seeder(MM):
    LCS = LocalCorrelationSeeder(neighborhood_size=3, keep_fraction=0.2)
    LCS.select_seeds(MM)
    assert LCS.next() == (9,)
    assert LCS.next() == (8,)
    assert LCS.next() is None
