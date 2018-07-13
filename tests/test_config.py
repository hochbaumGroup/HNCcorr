import pytest


@pytest.fixture
def c1():
    from hnccorr.config import Config

    return Config(patch_size=31, negative_seed_radius=10)


def test_config(c1):
    assert c1.patch_size == 31
    assert c1.negative_seed_radius == 10


def test_config_add(c1):
    from hnccorr.config import Config

    c2 = Config(patch_size=21, positive_seed_size=5)

    c3 = c1 + c2
    assert c1.patch_size == 31
    assert c2.patch_size == 21
    assert c3.patch_size == 21

    with pytest.raises(AttributeError):
        c2.negative_seed_radius
    assert c1.negative_seed_radius == 10
    assert c3.negative_seed_radius == 10

    with pytest.raises(AttributeError):
        c1.positive_seed_size == 5
    assert c2.positive_seed_size == 5
    assert c3.positive_seed_size == 5
