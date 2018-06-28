import pytest
import numpy as np


@pytest.fixture
def P(MM):
    from hnccorr.patch import Patch

    return Patch(MM, (5, ), 7, set([(4,), (5,), (6, )]))


@pytest.fixture
def P_boundary_max(MM):
    from hnccorr.patch import Patch

    return Patch(MM, (9, ), 7, set([(8,), (9, )]))


@pytest.fixture
def P_boundary_min(MM):
    from hnccorr.patch import Patch
    return Patch(MM, (1, ), 7, set([(0,), (1,), (2, )]))


def test_patch(P, MM):
    assert P.pixel_size == (7, )
    assert P.num_frames == 3
    assert P.positive_seeds == set([(2, ), (3, ), (4,)])
    assert P.coordinate_offset == (2,)

    np.testing.assert_equal(P[:], MM[:, 2:9])


def test_patch_boundary(P_boundary_min, P_boundary_max, MM):
    assert P_boundary_min.pixel_size == (7, )
    assert P_boundary_min.num_frames == 3
    assert P_boundary_min.positive_seeds == set([(0, ), (1, ), (2,)])

    np.testing.assert_equal(P_boundary_min[:], MM[:, :7])

    assert P_boundary_max.pixel_size == (7, )
    assert P_boundary_max.num_frames == 3
    assert P_boundary_max.positive_seeds == set([(5,), (6,)])

    np.testing.assert_equal(P_boundary_max[:], MM[:, 3:])
