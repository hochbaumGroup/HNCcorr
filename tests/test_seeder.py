import pytest

from hnccorr.utils import add_offset_set_coordinates


@pytest.fixture
def LCS(MM, PF):
    from hnccorr.seeder import LocalCorrelationSeeder

    return LocalCorrelationSeeder(
        MM, PF, positive_seed_size=3, neighborhood_size=3, keep_fraction=0.2
    )


def test_local_corr_seeder(LCS):
    patch = LCS.next()
    assert patch.positive_seeds == {(5,), (6,)}
    assert patch.coordinate_offset == (3,)

    patch = LCS.next()
    assert patch.positive_seeds == {(4,), (5,), (6,)}
    assert patch.coordinate_offset == (3,)

    assert LCS.next() is None
