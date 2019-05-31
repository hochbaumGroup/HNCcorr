import pytest

from hnccorr.hnccorr import HNCcorr
from hnccorr.candidate import Candidate


def test_candidate_segment(mocker, dummy):
    segmentations = "segmentations"
    best_segmentation = "best segmentation"
    pos_seed = "pos_seed"
    center_seed = 1

    segmentor = mocker.patch("hnccorr.hnc.HncParametric", autospec=True)(dummy, dummy)
    segmentor.solve.return_value = segmentations

    postprocessor = mocker.patch(
        "hnccorr.postprocessor.SizePostprocessor", autospec=True
    )(dummy, dummy, dummy)
    postprocessor.select.return_value = best_segmentation

    pos_seed_selector = mocker.patch(
        "hnccorr.seeds.PositiveSeedSelector", autospec=True
    )(dummy, dummy)
    pos_seed_selector.select.return_value = pos_seed

    hnccorr = HNCcorr(dummy, postprocessor, segmentor, pos_seed_selector, dummy)

    assert Candidate(center_seed, hnccorr).segment() == best_segmentation
    pos_seed_selector.select.assert_called_once_with(center_seed)
    segmentor.solve.assert_called_once_with(None, pos_seed, None)
    postprocessor.select.assert_called_once_with(segmentations)


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
