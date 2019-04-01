import pytest
import os
import numpy as np

from hnccorr.patch import Patch

TEST_DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "test_data"
)


@pytest.fixture
def MM():
    class MockMovie:
        def __init__(self):
            self.num_frames = 3
            self.pixel_size = (10,)
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
            self.pixel_size = self._A.shape[1:]

        def __getitem__(self, key):

            return self._A.__getitem__(key)

        def is_valid_pixel_index(self, index):
            return (
                index[0] >= 0
                and index[0] < 10
                and index[1] >= 0
                and index[1] < 10
            )

    return MockMovie()


@pytest.fixture
def pos_seeds():
    return {(4,), (5,), (6,)}


@pytest.fixture
def P(MM):
    return lambda x: Patch(MM, {}, x, 7)


@pytest.fixture
def P1(P):
    return P((5,))


@pytest.fixture
def P2(MM2):
    return Patch(MM2, {}, (5, 5), 3)
