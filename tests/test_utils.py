def test_add_offset():
    from hnccorr.utils import add_offset_coordinates

    assert add_offset_coordinates((1, 2), (3, 4)) == (4, 6)


def test_add_set_offset():
    from hnccorr.utils import add_offset_set_coordinates

    assert add_offset_set_coordinates({(0, 1), (1, 1)}, (2, 2)) == {
        (2, 3),
        (3, 3),
    }


def test_add_time_index():
    from hnccorr.utils import add_time_index

    assert add_time_index((5, 4)) == (slice(None, None), 5, 4)
