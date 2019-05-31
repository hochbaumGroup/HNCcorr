import pytest

from hnccorr.hnccorr import HNCcorr
from hnccorr.candidate import Candidate
from hnccorr.segmentation import Segmentation


@pytest.fixture
def H(seeder_fixed_val, postprocessor_select_first, segmentor_simple_segmentation):
    return HNCcorr(
        seeder_fixed_val,
        postprocessor_select_first,
        segmentor_simple_segmentation,
        "pos_seed_selector",
        "neg_seed_selector",
        "graph_constructor",
        "patch",
        "embedding",
    )


@pytest.fixture
def candidate(H):
    return Candidate(H.seeder.return_val, H)


def test_hnccorr_seeder(H, seeder_fixed_val):
    assert H.seeder == seeder_fixed_val


def test_hnccorr_movie(H, MM, seeder_fixed_val):
    seeder_fixed_val.called = True  # deactive seeder

    assert H.movie is None
    H.segment(MM)
    assert H.movie == MM


def test_hnccorr_seeder(H, seeder_fixed_val):
    assert H.seeder == seeder_fixed_val


def test_hnccorr_patch_class(H):
    assert H.patch_class == "patch"


def test_hnccorr_embedding_class(H):
    assert H.embedding_class == "embedding"


def test_hnccorr_positive_seed_selector(H):
    assert H.positive_seed_selector == "pos_seed_selector"


def test_hnccorr_negative_seed_selector(H):
    assert H.negative_seed_selector == "neg_seed_selector"


def test_hnccorr_graph_constructor(H):
    assert H.graph_constructor == "graph_constructor"


def test_hnccorr_seeder(H, postprocessor_select_first):
    assert H.postprocessor == postprocessor_select_first


def test_hnccorr_seeder(H, segmentor_simple_segmentation):
    assert H.segmentor == segmentor_simple_segmentation


def test_hnccorr_segmentations(H, MM, simple_segmentation):
    assert H.segmentations == []
    H.segment(MM)
    assert H.segmentations == [simple_segmentation]


def test_hnccorr_candidates(H, MM, candidate):
    assert H.candidates == []
    H.segment(MM)
    assert H.candidates == [candidate]


def test_hnccorr_reinitialize_candidates_for_movie(H, MM, candidate):
    H.segment(MM)
    assert H.candidates == [candidate]

    H.segment(MM)
    assert H.candidates == [candidate]


def test_hnccorr_reinitialize_segmentations_for_movie(
    H, MM, seeder_fixed_val, simple_segmentation
):
    H.segment(MM)
    assert H.segmentations == [simple_segmentation]

    H.segment(MM)
    assert H.segmentations == [simple_segmentation]
