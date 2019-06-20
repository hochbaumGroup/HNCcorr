import pytest

from hnccorr.base import HNCcorr, Candidate, HNCcorrConfig
from hnccorr.segmentation import Segmentation


@pytest.fixture
def mock_candidate_class(mocker):
    return mocker.patch("hnccorr.base.Candidate", autospec=True)


@pytest.fixture
def mock_seeder(mocker, dummy):
    return mocker.patch("hnccorr.seeds.LocalCorrelationSeeder", autospec=True)(
        dummy, dummy, dummy
    )


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
        "postprocessor",
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
def candidate(H):
    return Candidate(H.seeder.return_val, H)


class TestHnccorr:
    def test_hnccorr_seeder(self, H, seeder_fixed_val):
        assert H.seeder == seeder_fixed_val

    def test_hnccorr_from_config(self):
        assert isinstance(HNCcorr.from_config(HNCcorrConfig()), HNCcorr)

    def test_hnccorr_from_config(self):
        assert isinstance(HNCcorr.from_config(), HNCcorr)

    def test_hnccorr_from_non_default_config(self):
        assert isinstance(HNCcorr.from_config(HNCcorrConfig()), HNCcorr)

    def test_hnccorr_movie(self, H, MM, seeder_fixed_val):
        seeder_fixed_val.called = True  # deactive seeder

        assert H.movie is None
        H.segment(MM)
        assert H.movie == MM

    def test_hnccorr_patch_size(self, H):
        assert H.patch_size == "patch_size"

    def test_hnccorr_seeder(self, H, seeder_fixed_val):
        assert H.seeder == seeder_fixed_val

    def test_hnccorr_patch_class(self, H):
        assert H.patch_class == "patch"

    def test_hnccorr_embedding_class(self, H):
        assert H.embedding_class == "embedding"

    def test_hnccorr_segment_calls_candidate_segment(self, H, MM, mock_candidate_class):
        H.segment(MM)
        assert mock_candidate_class.return_value.segment.call_count > 0

    def test_hnccorr_positive_seed_selector(self, H):
        assert H.positive_seed_selector == "pos_seed_selector"

    def test_hnccorr_negative_seed_selector(self, H):
        assert H.negative_seed_selector == "neg_seed_selector"

    def test_hnccorr_graph_constructor(self, H):
        assert H.graph_constructor == "graph_constructor"

    def test_hnccorr_seeder(self, H, postprocessor_select_first):
        assert H.postprocessor == postprocessor_select_first

    def test_hnccorr_seeder(self, H, segmentor_simple_segmentation):
        assert H.segmentor == segmentor_simple_segmentation

    def test_hnccorr_segmentations_initialization(self, H):
        assert H.segmentations == []

    def test_hnccorr_candidates_initialization(self, H):
        assert H.candidates == []

    def test_hnccorr_candidates_after_segment(self, H, MM, mock_candidate_class):
        base_segmentation = Segmentation(set(), "segment")
        mock_candidate_class.return_value.segment.return_value = base_segmentation

        assert H.candidates == []
        H.segment(MM)
        assert len(H.candidates) == 1
        assert H.candidates[0].segment() == base_segmentation

        H.segment(MM)
        assert len(H.candidates) == 1
        assert H.candidates[0].segment() == base_segmentation

    def test_hnccorr_segmentations_after_segment(self, H, MM, mock_candidate_class):
        base_segmentation = Segmentation(set(), "segment")
        mock_candidate_class.return_value.segment.return_value = base_segmentation

        H.segment(MM)
        assert H.segmentations == [base_segmentation]

        H.segment(MM)
        assert H.segmentations == [base_segmentation]

    def test_hnccorr_segment_none_is_not_added_to_segmentations(
        self, MM, H, mock_candidate_class
    ):
        mock_candidate_class.return_value.segment.return_value = None

        H.segment(MM)

        assert H.segmentations == []

    def test_hnccor_exclude_previously_segmented_pixels(
        self, mock_seeder, dummy, mock_candidate_class
    ):
        mock_seeder.next.side_effect = ("seed1", None)
        mock_candidate_class.return_value.segment.return_value = Segmentation(
            {"pixel"}, 1
        )
        H = HNCcorr(
            mock_seeder,
            "postprocessor",
            "mock_segmentor",
            "pos_seed_selector",
            "neg_seed_selector",
            "graph_constructor",
            mock_candidate_class,
            "patch",
            "embedding",
            "patch_size",
        )

        H.segment(dummy)

        mock_seeder.exclude_pixels.assert_called_once_with({"pixel"})
