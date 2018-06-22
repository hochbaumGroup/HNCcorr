import pytest


@pytest.fixture
def PP():
    from hnccorr.postprocessor import SizePostprocessor

    return SizePostprocessor(2, 10, 3)


def test_size_postprocessor_select(PP, S1, S2):
    assert PP.select([S1, S2]) == S1


def test_size_postprocessor_select_no_candidates(PP, S2, S3):
    assert PP.select([S2, S3]) is None
