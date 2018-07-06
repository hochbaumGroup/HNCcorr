import pytest
import numpy as np
from copy import copy, deepcopy


@pytest.fixture
def P(MM):
    from hnccorr.patch import Patch

    return Patch(MM, (5,), 7, 2, {(4,), (5,), (6,)})


@pytest.fixture
def PF(MM):
    from hnccorr.patch import PatchFactory

    return PatchFactory(MM, 7, 2)


@pytest.fixture
def P_boundary(MM, request):
    from hnccorr.patch import Patch

    return Patch(MM, request.param, 7, 3, {(5,)})


def test_pixel_size(P):
    assert P.pixel_size == (7,)


def test_num_frames(P):
    assert P.num_frames == 3


def test_seeds(P):
    assert P.positive_seeds == {(2,), (3,), (4,)}
    assert P.negative_seeds == {(1,), (5,)}


def test_data(P, MM):
    np.testing.assert_equal(P[:], MM[:, 2:9])


@pytest.mark.parametrize(
    "P_boundary, offset", ([(1,), (0,)], [(9,), (3,)]), indirect=["P_boundary"]
)
def test_offset(P_boundary, offset):
    assert P_boundary.coordinate_offset == offset


def test_patch_equal(P):
    assert P == copy(P)
    assert P != deepcopy(P)


def test_patch_factory(PF, MM, P):
    patch = PF.construct((5,), {(4,), (5,), (6,)})
    assert patch == P
