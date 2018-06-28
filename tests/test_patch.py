import pytest
import numpy as np


@pytest.fixture
def P(MM):
    from hnccorr.patch import Patch

    return Patch(MM, (5, ), 7, set([(4,), (5,), (6, )]))


def test_patch(P, MM):
    assert P.pixel_size == (7, )
    assert P.num_frames == 3
    assert P.positive_seeds == set([(2, ), (3, ), (4,)])

    np.testing.assert_equal(P[:], MM[:, 2:9])
