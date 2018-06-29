import pytest

from hnccorr.utils import add_offset_set_coordinates


@pytest.fixture
def LCS(MM):
    from hnccorr.seeder import LocalCorrelationSeeder

    return LocalCorrelationSeeder(
        MM,
        positive_seed_size=3,
        neighborhood_size=3,
        keep_fraction=0.2,
        window_size=7,
    )


def test_local_corr_seeder(LCS):
    patch = LCS.next()
    assert add_offset_set_coordinates(
        patch.positive_seeds, patch.coordinate_offset
    ) == set([(8,), (9,)])

    patch = LCS.next()
    assert add_offset_set_coordinates(
        patch.positive_seeds, patch.coordinate_offset
    ) == set([(7,), (8,), (9,)])

    assert LCS.next() is None
