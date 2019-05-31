import pytest

from hnccorr.hnccorr import HNCcorr
from hnccorr.candidate import Candidate


def test_candidate_segment(mocker, dummy, simple_segmentation):
    segmentor = mocker.patch("hnccorr.hnc.HncParametric", autospec=True)(dummy, dummy)
    segmentor.solve.return_value = "segmentations"
    postprocessor = mocker.patch(
        "hnccorr.postprocessor.SizePostprocessor", autospec=True
    )(dummy, dummy, dummy)
    postprocessor.select.return_value = simple_segmentation

    hnccorr = HNCcorr(dummy, postprocessor, segmentor, dummy, dummy)

    assert Candidate(1, hnccorr).segment() == simple_segmentation
    segmentor.solve.assert_called_once_with(None, None, None)
    postprocessor.select.assert_called_once_with("segmentations")


def test_candidate_equality():
    assert Candidate(1, "a") == Candidate(1, "a")
    assert Candidate(1, "a") != Candidate(2, "a")
    assert Candidate(1, "a") != Candidate(1, "b")


def test_candidate_segmentations(simple_candidate, simple_segmentation):
    assert simple_candidate.segmentations is None
    simple_candidate.segment()
    assert simple_candidate.segmentations == [simple_segmentation]


def test_candidate_best_segmentations(simple_candidate, simple_segmentation):
    assert simple_candidate.best_segmentation is None
    simple_candidate.segment()
    assert simple_candidate.best_segmentation == simple_segmentation
