import pytest


@pytest.fixture
def PP():
    from hnccorr.postprocessor import SizePostprocessor

    return SizePostprocessor(2, 10, 3)


@pytest.fixture
def S3(S):
    return S(
        {
            (0, 0),
            (1, 0),
            (2, 0),
            (0, 1),
            (2, 1),
            (0, 2),
            (1, 2),
            (2, 2),
            (3, 0),
            (3, 1),
            (3, 2),
        }
    )


def test_size_postprocessor_select(PP, S1, S2):
    assert PP.select([S1, S2]) == S1


def test_size_postprocessor_select_no_candidates(PP, S2, S3):
    assert PP.select([S2, S3]) is None
