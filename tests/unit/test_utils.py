# Copyright © 2017. Regents of the University of California (Regents). All Rights
# Reserved.
#
# Permission to use, copy, modify, and distribute this software and its documentation
# for educational, research, and not-for-profit purposes, without fee and without a
# signed licensing agreement, is hereby granted, provided that the above copyright
# notice, this paragraph and the following two paragraphs appear in all copies,
# modifications, and distributions. Contact The Office of Technology Licensing, UC
# Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201,
# for commercial licensing opportunities. Created by Quico Spaen, Roberto Asín-Achá,
# and Dorit S. Hochbaum, Department of Industrial Engineering and Operations Research,
# University of California, Berkeley.
#
# IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
# INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE
# OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE
# SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS
# IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES,
# ENHANCEMENTS, OR MODIFICATIONS.
import os

from conftest import TEST_DATA_DIR

from hnccorr.utils import (
    four_neighborhood,
    generate_pixels,
    add_offset_to_coordinate,
    add_offset_set_coordinates,
    add_time_index,
    list_images,
    eight_neighborhood,
)


def test_add_offset():
    assert add_offset_to_coordinate((1, 2), (3, 4)) == (4, 6)


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
