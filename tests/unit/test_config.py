import pytest

from hnccorr.config import HNCcorrConfig, default_config


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


def test_default_config():
    assert default_config.seeder_mask_size == 3
    assert default_config.percentage_of_seeds == pytest.approx(0.4)
    assert default_config.postprocessor_min_cell_size == 40
    assert default_config.postprocessor_max_cell_size == 200
    assert default_config.postprocessor_preferred_cell_size == 80
    assert default_config.positive_seed_radius == 0
    assert default_config.negative_seed_circle_radius == pytest.approx(10.0)
    assert default_config.negative_seed_circle_count == 10
    assert default_config.gaussian_similarity_alpha == pytest.approx(1.0)
    assert default_config.sparse_computation_grid_distance == pytest.approx(1 / 35.0)
    assert default_config.sparse_computation_dimension == 3
    assert default_config.patch_size == 31
