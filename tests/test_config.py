def test_configuration():
    from hnccorr.config import Config

    c = Config(patch_size=31, negative_seed_radius=10)

    assert c.patch_size == 31
    assert c.negative_seed_radius == 10
