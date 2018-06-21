import pytest
# Temporary hardcopy from segmentation
# TODO: Creat fixture file


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

    return Segmentation(P, set([(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (1, 2), (2, 2), (3, 0), (3, 1), (3, 2)]), 0.75)


@pytest.fixture
def PP():
    from hnccorr.postprocessor import SizePostprocessor

    return SizePostprocessor(2, 10, 3)


def test_size_postprocessor_select(PP, S1, S2):
    assert PP.select([S1, S2]) == S1


def test_size_postprocessor_select_no_candidates(PP, S1, S2, S3):
    assert PP.select([S2, S3]) is None
