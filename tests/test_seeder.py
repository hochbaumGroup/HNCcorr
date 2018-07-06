import pytest

from hnccorr.utils import add_offset_set_coordinates


@pytest.fixture
def LCS(MM, MPF):
    from hnccorr.seeder import LocalCorrelationSeeder

    return LocalCorrelationSeeder(
        MM, MPF, positive_seed_size=3, neighborhood_size=3, keep_fraction=0.2
    )


def test_local_corr_seeder(LCS):
    patch = LCS.next()
    assert patch["positive_seeds"] == {(8,), (9,)}

    patch = LCS.next()
    assert patch["positive_seeds"] == {(7,), (8,), (9,)}

    assert LCS.next() is None
