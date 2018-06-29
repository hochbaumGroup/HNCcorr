import pytest


@pytest.fixture
def MM():
    import numpy as np

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
def MP():
    class MockPatch(object):
        def __init__(self):
            self.shape = (5, 5)
            self.positive_seeds = {(1, 0)}

    return MockPatch()


@pytest.fixture
def MPF(MM, MP):
    class MockPatchFactory:
        def construct(self, center_seed, positive_seeds):
            return {
                "center_seed": center_seed,
                "positive_seeds": positive_seeds,
            }

    return MockPatchFactory()


@pytest.fixture
def S1(MP):
    from hnccorr.segmentation import Segmentation

    return Segmentation(
        MP,
        {(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (1, 2), (2, 2)},
        0.5,
    )


@pytest.fixture
def S2(MP):
    from hnccorr.segmentation import Segmentation

    return Segmentation(MP, {(1, 0)}, 0.5)


@pytest.fixture
def S3(MP):
    from hnccorr.segmentation import Segmentation

    return Segmentation(
        MP,
        {
            (0, 0),
            (1, 0),
            (2, 0),
            (0, 1),
            (2, 1),
            (0, 2),
            (1, 2),
            (2, 2),
            (3, 0),
            (3, 1),
            (3, 2),
        },
        0.75,
    )
