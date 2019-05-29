import pytest

from hnccorr.hnccorr import HNCcorr
from hnccorr.candidate import Candidate
from hnccorr.segmentation import Segmentation


@pytest.fixture
def mock_seeder():
    class MockSeeder:
        def __init__(self):
            self.called = False
            self.return_val = 1

        def select_seeds(self, movie):
            pass

        def next(self):
            if self.called:
                return None
            else:
                self.called = True
                return self.return_val

        def reset(self):
            self.called = False

    return MockSeeder()


@pytest.fixture
def postprocessor_select_first():
    class MockPostProcessor:
        def segment(self, segmentations):
            return segmentations[0]

    return MockPostProcessor()


@pytest.fixture
def H(mock_seeder, postprocessor_select_first):
    return HNCcorr(mock_seeder, postprocessor_select_first)


def test_hnccorr_segmentations(H, MM, simple_segmentation):
    assert H.segmentations == []
    H.segment(MM)
    assert H.segmentations == [simple_segmentation]


def test_hnccorr_candidates(H, MM, mock_seeder):
    assert H.candidates == []
    H.segment(MM)
    assert H.candidates == [Candidate(mock_seeder.return_val)]


def test_hnccorr_reinitialize_candidates_for_movie(H, MM, mock_seeder):
    H.segment(MM)
    assert H.candidates == [Candidate(mock_seeder.return_val)]

    H.segment(MM)
    assert H.candidates == [Candidate(mock_seeder.return_val)]


def test_hnccorr_reinitialize_segmentations_for_movie(
    H, MM, mock_seeder, simple_segmentation
):
    H.segment(MM)
    assert H.segmentations == [simple_segmentation]

    H.segment(MM)
    assert H.segmentations == [simple_segmentation]
