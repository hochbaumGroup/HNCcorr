import pytest
import numpy as np
from copy import copy, deepcopy
from pytest_mock import mocker


def test_even_windowsize(MM, pos_seeds):
    from hnccorr.patch import Patch

    with pytest.raises(ValueError):
        Patch(MM, {}, (5,), 6, 2)


def test_pixel_size(P1):
    assert P1.pixel_size == (7,)


def test_num_frames(P1):
    assert P1.num_frames == 3


def test_seeds(P1):
    assert P1.seeds.positive_seeds == {(2,), (3,), (4,)}
    assert P1.seeds.negative_seeds == {(1,), (5,)}


def test_data(P1, MM):
    np.testing.assert_equal(P1[:], MM[:, 2:9])


@pytest.mark.parametrize("center_seed, offset", ([(1,), (0,)], [(9,), (3,)]))
def test_offset(P, center_seed, offset):
    assert P(center_seed).coordinate_offset == offset


def test_patch_equal(P1):
    assert P1 == copy(P1)
    assert P1 != deepcopy(P1)


def test_segment(mocker, P1, S4):
    from hnccorr.edge_selection import SparseComputation

    mocker.patch.object(SparseComputation, "select_edges")
    SparseComputation.select_edges.return_value = []

    assert P1.segment() == [S4]


def test_patch_factory(PF, MM, P1, pos_seeds):
    patch = PF.construct((5,), pos_seeds)
    assert patch == P1
