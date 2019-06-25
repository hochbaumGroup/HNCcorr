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
import pytest
import os
import numpy as np

from hnccorr.movie import Patch
from hnccorr.segmentation import Segmentation
from hnccorr.base import Candidate

TEST_DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../test_data"
)


@pytest.fixture
def dummy():
    return "Dummy"


@pytest.fixture
def mock_movie(mocker, dummy):
    movie = mocker.patch("hnccorr.movie.Movie", autospec=True)(dummy, dummy)
    return movie


@pytest.fixture
def MM(mock_movie):
    mock_movie.num_frames = 3
    mock_movie.pixel_shape = (10,)
    mock_movie.num_dimensions = 1

    def data(self, key):
        A = np.zeros((3, 10))
        A[0, :] = 1
        A[1, :] = -1
        A[2, :] = np.arange(10) / 10
        return A.__getitem__(key)

    mock_movie.__getitem__ = data
    mock_movie.is_valid_pixel_coordinate = lambda x: x[0] >= 0 and x[0] < 10

    return mock_movie


@pytest.fixture
def MM2():
    class MockMovie:
        def __init__(self):
            self.num_dimensions = 2
            self.num_frames = 1

            self._A = np.zeros((2, 10, 10))
            self.pixel_shape = self._A.shape[1:]

        def __getitem__(self, key):

            return self._A.__getitem__(key)

        def is_valid_pixel_index(self, index):
            return index[0] >= 0 and index[0] < 10 and index[1] >= 0 and index[1] < 10

    return MockMovie()


@pytest.fixture
def pos_seeds():
    return {(4,), (5,), (6,)}


@pytest.fixture
def simple_segmentation():
    return Segmentation({(0, 1)}, 1.0)


@pytest.fixture
def segmentor_simple_segmentation(simple_segmentation):
    class MockSegmentor:
        def solve(self, graph, pos_seeds, neg_seeds):
            return [simple_segmentation]

    return MockSegmentor()


@pytest.fixture
def simple_positive_seed_selector():
    class MockPositiveSeedSelector:
        def select(center_seed):
            return center_seed

    return MockPositiveSeedSelector()


@pytest.fixture
def postprocessor_select_first():
    class MockPostProcessor:
        def select(self, segmentations):
            return segmentations[0]

    return MockPostProcessor()


@pytest.fixture
def seeder_fixed_val():
    class MockSeeder:
        def __init__(self):
            self.called = False
            self.return_val = 1

        def select_seeds(self, movie):
            pass

        def exclude_pixels(self, pixel_set):
            pass

        def next(self):
            if self.called:
                return None
            else:
                self.called = True
                return self.return_val

        def reset(self):
            self.called = False

    return MockSeeder()
