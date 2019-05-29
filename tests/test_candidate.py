import pytest
from hnccorr.candidate import Candidate
from hnccorr.segmentation import Segmentation


def test_candidate_segment(simple_segmentation):
    c = Candidate(1)
    best_segmentation = c.segment()

    assert best_segmentation == simple_segmentation
