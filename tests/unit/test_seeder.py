import pytest

from hnccorr.seeder import LocalCorrelationSeeder


@pytest.fixture
def LCS():
    return LocalCorrelationSeeder(3, 0.2, 2)


def test_local_corr_seeder(LCS, MM):
    LCS.select_seeds(MM)
    assert LCS.next() == (9,)
    assert LCS.next() == (8,)
    assert LCS.next() is None


def test_local_corr_seeder_reset(LCS, MM):
    LCS.select_seeds(MM)
    assert LCS.next() == (9,)

    LCS.reset()
    assert LCS.next() == (9,)


def test_seeder_exclude_pixels(LCS, MM):
    LCS.select_seeds(MM)
    assert LCS.next() == (9,)
    LCS.exclude_pixels({(8,)})
    assert LCS.next() is None


def test_seeder_exclude_pixels_boundary(LCS, MM):
    LCS.select_seeds(MM)
    assert LCS.next() == (9,)
    LCS.exclude_pixels({(6,)})
    assert LCS.next() is None


def test_seeder_reset_excluded_pixels(LCS, MM):
    LCS.select_seeds(MM)
    assert LCS.next() == (9,)
    LCS.exclude_pixels({(9,)})
    LCS.reset()
    assert LCS.next() == (9,)


def test_seeder_select_seeds_should_reset_excluded_pixels(LCS, MM):
    LCS.select_seeds(MM)
    LCS.exclude_pixels({(9,)})
    LCS.select_seeds(MM)
    assert LCS.next() == (9,)
