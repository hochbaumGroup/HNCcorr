import pytest
from hnccorr.candidate import Candidate
from hnccorr.segmentation import Segmentation


@pytest.fixture
def simple_candidate():
    return Candidate(1)


def test_candidate_segment(simple_candidate, simple_segmentation):
    assert simple_candidate.segment() == simple_segmentation


def test_candidate_segmentations(simple_candidate, simple_segmentation):
    simple_candidate.segment()
    assert simple_candidate.segmentations == [simple_segmentation]
