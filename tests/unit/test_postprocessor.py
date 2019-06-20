import pytest
from hnccorr.postprocessor import SizePostprocessor
from hnccorr.segmentation import Segmentation


@pytest.fixture
def postprocessor():
    return SizePostprocessor(2, 5, 3)


def test_select_preference(postprocessor):
    S1 = Segmentation({(0, 1), (0, 2), (0, 3)}, 0.5)
    S2 = Segmentation({(0, 1), (0, 2)}, 0.5)
    assert postprocessor.select([S1, S2]) == S1


def test_select_no_valid_candidates(postprocessor):
    assert (
        postprocessor.select(
            [
                Segmentation({(0, 1)}, 0.5),
                Segmentation({(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2)}, 0.5),
            ]
        )
        is None
    )
