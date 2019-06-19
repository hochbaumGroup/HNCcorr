import pytest
import os
import numpy as np

from hnccorr.movie import Patch
from hnccorr.segmentation import Segmentation
from hnccorr.candidate import Candidate

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data")


@pytest.fixture
def dummy():
    return "Dummy"


@pytest.fixture
def MM():
    class MockMovie:
        def __init__(self):
            self.num_frames = 3
            self.pixel_shape = (10,)
            self.num_dimensions = 1

        def __getitem__(self, key):
            A = np.zeros((3, 10))
            A[0, :] = 1
            A[1, :] = -1
            A[2, :] = np.arange(10) / 10
            return A.__getitem__(key)

        def is_valid_pixel_index(self, index):
            return index[0] >= 0 and index[0] < 10

    return MockMovie()


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
