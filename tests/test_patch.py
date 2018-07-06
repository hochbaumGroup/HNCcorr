import pytest
import numpy as np


@pytest.fixture
def P(MM):
    from hnccorr.patch import Patch

    return Patch(MM, (5,), 7, 2, {(4,), (5,), (6,)})


@pytest.fixture
def PF(MM):
    from hnccorr.patch import PatchFactory

    return PatchFactory(MM, 7, 2)


@pytest.fixture
def P_boundary_max(MM):
    from hnccorr.patch import Patch

    return Patch(MM, (9,), 7, 3, {(8,), (9,)})


@pytest.fixture
def P_boundary_min(MM):
    from hnccorr.patch import Patch

    return Patch(MM, (1,), 7, 2, {(0,), (1,), (2,)})


def test_pixel_size(P):
    assert P.pixel_size == (7,)


def test_num_frames(P):
    assert P.num_frames == 3


def test_seeds(P):
    assert P.positive_seeds == {(2,), (3,), (4,)}
    assert P.negative_seeds == {(1,), (5,)}


def test_offset(P):
    assert P.coordinate_offset == (2,)


def test_data(P, MM):
    np.testing.assert_equal(P[:], MM[:, 2:9])


def test_patch_boundary(P_boundary_min, P_boundary_max, MM):
    assert P_boundary_min.pixel_size == (7,)
    assert P_boundary_min.num_frames == 3
    assert P_boundary_min.positive_seeds == {(0,), (1,), (2,)}

    np.testing.assert_equal(P_boundary_min[:], MM[:, :7])

    assert P_boundary_max.pixel_size == (7,)
    assert P_boundary_max.num_frames == 3
    assert P_boundary_max.positive_seeds == {(5,), (6,)}
    assert P_boundary_max.negative_seeds == {(3,)}

    np.testing.assert_equal(P_boundary_max[:], MM[:, 3:])


def test_patch_factory(PF, MM, P):
    patch = PF.construct((5,), {(4,), (5,), (6,)})
    assert patch == P
