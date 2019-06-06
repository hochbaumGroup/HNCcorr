import pytest

from hnccorr.config import HNCcorrConfig


@pytest.fixture
def config():
    return HNCcorrConfig(patch_size=31, negative_seed_radius=10)


def test_attributes(config):
    assert config.patch_size == 31
    assert config.negative_seed_radius == 10


def test_add_config(config):
    config2 = HNCcorrConfig(patch_size=21, positive_seed_size=5)

    config3 = config + config2
    assert config.patch_size == 31
    assert config2.patch_size == 21
    assert config3.patch_size == 21

    with pytest.raises(AttributeError):
        config2.negative_seed_radius
    assert config.negative_seed_radius == 10
    assert config3.negative_seed_radius == 10

    with pytest.raises(AttributeError):
        config.positive_seed_size == 5
    assert config2.positive_seed_size == 5
    assert config3.positive_seed_size == 5
