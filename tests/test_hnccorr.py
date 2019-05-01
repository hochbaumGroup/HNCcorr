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

    return MockSeeder()


def test_hnccorr_segment(MM, mock_seeder):
    h = HNCcorr(mock_seeder)
    h.segment(MM)
    assert h.candidates == [Candidate(mock_seeder.return_val)]


def test_hnccorr_segmentations(MM, mock_seeder):
    h = HNCcorr(mock_seeder)
    assert h.segmentations == []
    h.segment(MM)
    assert h.segmentations == [Segmentation({(0, 1)}, 1.0)]
