import pytest


@pytest.fixture
def P():
    class Mock_Patch(object):
        def __init__(self):
            self.shape = (5, 5)
            self.positive_seeds = set([(1, 0)])

    return Mock_Patch()


@pytest.fixture
def S1(P):
    from hnccorr.segmentation import Segmentation

    return Segmentation(
        P,
        set([(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (1, 2), (2, 2)]),
        0.5,
    )


@pytest.fixture
def S2(P):
    from hnccorr.segmentation import Segmentation

    return Segmentation(P, set([(1, 0)]), 0.5)


@pytest.fixture
def S3(P):
    from hnccorr.segmentation import Segmentation

    return Segmentation(
        P,
        set(
            [
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
            ]
        ),
        0.75,
    )
