import pytest

from hnccorr.hnccorr import HNCcorr
from hnccorr.candidate import Candidate
from hnccorr.segmentation import Segmentation
from hnccorr.config import HNCcorrConfig


@pytest.fixture
def mock_candidate_class(mocker):
    return mocker.patch("hnccorr.candidate.Candidate", autospec=True)


@pytest.fixture
def H(
    dummy,
    seeder_fixed_val,
    postprocessor_select_first,
    segmentor_simple_segmentation,
    mock_candidate_class,
):
    return HNCcorr(
        seeder_fixed_val,
        postprocessor_select_first,
        segmentor_simple_segmentation,
        "pos_seed_selector",
        "neg_seed_selector",
        "graph_constructor",
        mock_candidate_class,
        "patch",
        "embedding",
        "patch_size",
    )


@pytest.fixture
def hnccorr_with_mocked_postprocessor(
    seeder_fixed_val, mock_postprocessor, mock_candidate_class
):
    return HNCcorr(
        seeder_fixed_val,
        mock_postprocessor,
        "segmentor",
        "pos_seed_selector",
        "neg_seed_selector",
        "graph_constructor",
        mock_candidate_class,
        "patch",
        "embedding",
        "patch_size",
    )


@pytest.fixture
def candidate(H):
    return Candidate(H.seeder.return_val, H)


def test_hnccorr_seeder(H, seeder_fixed_val):
    assert H.seeder == seeder_fixed_val


def test_hnccorr_from_config():
    assert isinstance(HNCcorr.from_config(HNCcorrConfig()), HNCcorr)


def test_hnccorr_movie(H, MM, seeder_fixed_val):
    seeder_fixed_val.called = True  # deactive seeder

    assert H.movie is None
    H.segment(MM)
    assert H.movie == MM


def test_hnccorr_patch_size(H):
    assert H.patch_size == "patch_size"


def test_hnccorr_seeder(H, seeder_fixed_val):
    assert H.seeder == seeder_fixed_val


def test_hnccorr_patch_class(H):
    assert H.patch_class == "patch"


def test_hnccorr_embedding_class(H):
    assert H.embedding_class == "embedding"


def test_hnccorr_segment_calls_candidate_segment(H, MM, mock_candidate_class):
    mock_candidate_class.return_value.segment.return_value = ["segmentation"]
    H.segment(MM)
    assert mock_candidate_class.return_value.segment.call_count > 0


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


def test_hnccorr_segmentations_initialization(H):
    assert H.segmentations == []


def test_hnccorr_candidates_initialization(H):
    assert H.candidates == []


def test_hnccorr_candidates_after_segment(H, MM, mock_candidate_class):
    mock_candidate_class.return_value.segment.return_value = "segment"

    assert H.candidates == []
    H.segment(MM)
    assert len(H.candidates) == 1
    assert H.candidates[0].segment() == "segment"

    H.segment(MM)
    assert len(H.candidates) == 1
    assert H.candidates[0].segment() == "segment"


def test_hnccorr_candidates_after_segment(H, MM, mock_candidate_class):
    mock_candidate_class.return_value.segment.return_value = ["segment"]
    H.segment(MM)
    assert H.segmentations == ["segment"]

    H.segment(MM)
    assert H.segmentations == ["segment"]


@pytest.fixture
def mock_postprocessor(mocker, dummy):
    return mocker.patch("hnccorr.postprocessor.SizePostprocessor", autospec=True)(
        dummy, dummy, dummy
    )


def test_hnccorr_segment_calls_postprocessor_select(
    MM, mock_candidate_class, hnccorr_with_mocked_postprocessor, mock_postprocessor
):
    mock_candidate_class.return_value.segment.return_value = ["segment", "segment2"]

    hnccorr_with_mocked_postprocessor.segment(MM)

    mock_postprocessor.select.assert_called_once_with(["segment", "segment2"])


def test_hnccorr_segment_segmentations(
    MM, hnccorr_with_mocked_postprocessor, mock_postprocessor
):
    mock_postprocessor.select.return_value = "first_choice"

    hnccorr_with_mocked_postprocessor.segment(MM)

    assert hnccorr_with_mocked_postprocessor.segmentations == ["first_choice"]


def test_hnccorr_segment_none_is_not_added_to_segmentations(
    MM, hnccorr_with_mocked_postprocessor, mock_postprocessor
):
    mock_postprocessor.select.return_value = None

    hnccorr_with_mocked_postprocessor.segment(MM)

    assert hnccorr_with_mocked_postprocessor.segmentations == []
