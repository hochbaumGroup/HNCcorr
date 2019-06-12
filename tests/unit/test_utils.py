import os

from conftest import TEST_DATA_DIR

from hnccorr.utils import (
    four_neighborhood,
    generate_pixels,
    add_offset_coordinates,
    add_offset_set_coordinates,
    add_time_index,
    list_images,
    fill_holes,
    eight_neighborhood,
)


def test_add_offset():
    assert add_offset_coordinates((1, 2), (3, 4)) == (4, 6)


def test_add_set_offset():
    assert add_offset_set_coordinates({(0, 1), (1, 1)}, (2, 2)) == {(2, 3), (3, 3)}


def test_add_time_index():
    assert add_time_index((5, 4)) == (slice(None, None), 5, 4)


def test_list_images():
    images = list_images(TEST_DATA_DIR)
    expected_images = map(
        lambda x: os.path.join("./test_data/simple_movie", x),
        ["simple_movie00000.tif", "simple_movie00001.tif", "simple_movie00002.tif"],
    )

    for i, e in zip(images, expected_images):
        assert os.path.abspath(i) == os.path.abspath(e)


def test_fill_holes():
    assert fill_holes({(1,), (3,)}, (5,)) == {(1,), (2,), (3,)}


def test_eight_neighborhood():
    assert eight_neighborhood(1, 1) == {(-1,), (0,), (1,)}
    assert eight_neighborhood(1, 2) == {(-2,), (-1,), (0,), (1,), (2,)}
    assert eight_neighborhood(2, 1) == {
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 0),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    }


def test_four_neighborhood():
    assert four_neighborhood(1) == {(-1,), (0,), (1,)}
    assert four_neighborhood(2) == {(-1, 0), (0, 0), (1, 0), (0, -1), (0, 1)}


def test_generate_pixles():
    assert set(generate_pixels((1, 2))) == {(0, 0), (0, 1)}
