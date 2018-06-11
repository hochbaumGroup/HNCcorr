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
def SS(S1, S2):
    from hnccorr.segmentation import Segmentations

    return Segmentations([S1, S2], 2, 10, 3)


def test_segmentation_clean(S1):
    S1.clean()
    assert S1.selection == set(
        [
            (0, 0),
            (1, 0),
            (2, 0),
            (0, 1),
            (1, 1),
            (2, 1),
            (0, 2),
            (1, 2),
            (2, 2),
        ]
    )


def test_segmentations_clean(SS, S1):
    SS.clean()
    S1.clean()
    assert SS.select().selection == S1.selection


def test_best_segmentation(SS, S1):
    assert SS.select().weight == S1.weight
