import pytest

from hnccorr.hnccorr import HNCcorr
from hnccorr.candidate import Candidate
from hnccorr.segmentation import Segmentation


@pytest.fixture
def H(seeder_fixed_val, postprocessor_select_first, segmentor_simple_segmentation):
    return HNCcorr(
        seeder_fixed_val, postprocessor_select_first, segmentor_simple_segmentation
    )


@pytest.fixture
def candidate(H):
    return Candidate(H._seeder.return_val, H)


def test_hnccorr_segmentations(H, MM, simple_segmentation):
    assert H.segmentations == []
    H.segment(MM)
    assert H.segmentations == [simple_segmentation]


def test_hnccorr_candidates(H, MM, candidate):
    assert H.candidates == []
    H.segment(MM)
    assert H.candidates == [candidate]


def test_hnccorr_reinitialize_candidates_for_movie(H, MM, candidate):
    H.segment(MM)
    assert H.candidates == [candidate]

    H.segment(MM)
    assert H.candidates == [candidate]


def test_hnccorr_reinitialize_segmentations_for_movie(
    H, MM, seeder_fixed_val, simple_segmentation
):
    H.segment(MM)
    assert H.segmentations == [simple_segmentation]

    H.segment(MM)
    assert H.segmentations == [simple_segmentation]
