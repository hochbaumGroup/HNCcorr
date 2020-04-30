# Copyright © 2017. Regents of the University of California (Regents). All Rights
# Reserved.
#
# Permission to use, copy, modify, and distribute this software and its documentation
# for educational, research, and not-for-profit purposes, without fee and without a
# signed licensing agreement, is hereby granted, provided that the above copyright
# notice, this paragraph and the following two paragraphs appear in all copies,
# modifications, and distributions. Contact The Office of Technology Licensing, UC
# Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201,
# for commercial licensing opportunities. Created by Quico Spaen, Roberto Asín-Achá,
# and Dorit S. Hochbaum, Department of Industrial Engineering and Operations Research,
# University of California, Berkeley.
#
# IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
# INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE
# OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE
# SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS
# IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES,
# ENHANCEMENTS, OR MODIFICATIONS.
import pytest
from copy import deepcopy

from hnccorr.base import HNCcorr, Candidate, HNCcorrConfig, DEFAULT_CONFIG
from hnccorr.segmentation import Segmentation


@pytest.fixture
def config():
    return HNCcorrConfig(patch_size=31, negative_seed_circle_radius=10)


@pytest.fixture
def candidate(H):
    return Candidate(H.seeder.return_val, H)


@pytest.fixture
def mock_candidate_class(mocker):
    return mocker.patch("hnccorr.base.Candidate", autospec=True)


@pytest.fixture
def mock_seeder(mocker, dummy):
    return mocker.patch("hnccorr.seeds.LocalCorrelationSeeder", autospec=True)(
        dummy, dummy, dummy, dummy
    )


@pytest.fixture
def mock_pos_seed_selector(mocker, dummy):
    pos_seed_selector = mocker.patch(
        "hnccorr.seeds.PositiveSeedSelector", autospec=True
    )(dummy)

    return pos_seed_selector


@pytest.fixture
def mock_neg_seed_selector(mocker, dummy):
    neg_seed_selector = mocker.patch(
        "hnccorr.seeds.NegativeSeedSelector", autospec=True
    )(dummy, dummy)
    return neg_seed_selector


@pytest.fixture
def mock_segmentor(mocker, dummy):
    segmentor = mocker.patch(
        "hnccorr.segmentation.HncParametricWrapper", autospec=True
    )(dummy, dummy)
    return segmentor


@pytest.fixture
def mock_postprocessor(mocker, dummy):
    postprocessor = mocker.patch(
        "hnccorr.postprocessor.SizePostprocessor", autospec=True
    )(dummy, dummy, dummy)
    return postprocessor


@pytest.fixture
def mock_graph_constructor(mocker, dummy):
    graph_constructor = mocker.patch("hnccorr.graph.GraphConstructor", autospec=True)(
        dummy, dummy
    )
    return graph_constructor


@pytest.fixture
def mock_patch_class(mocker, dummy):
    patch_class = mocker.patch("hnccorr.movie.Patch", autospec=True)
    return patch_class


@pytest.fixture
def mock_embedding_class(mocker, dummy):
    embedding_class = mocker.patch("hnccorr.graph.CorrelationEmbedding", autospec=True)
    return embedding_class


@pytest.fixture
def mock_segmentation_class(mocker):
    return mocker.patch("hnccorr.segmentation.Segmentation", autospec=True)


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
def hnccorr(
    dummy,
    mock_movie,
    mock_postprocessor,
    mock_segmentor,
    mock_pos_seed_selector,
    mock_neg_seed_selector,
    mock_graph_constructor,
    mock_patch_class,
    mock_embedding_class,
):
    H = HNCcorr(
        dummy,
        mock_postprocessor,
        mock_segmentor,
        mock_pos_seed_selector,
        mock_neg_seed_selector,
        mock_graph_constructor,
        Candidate,
        mock_patch_class,
        mock_embedding_class,
        "patch_size",
    )

    H.movie = mock_movie
    return H


class TestCandidate:
    def test_candidate_segment(
        self,
        hnccorr,
        dummy,
        mock_movie,
        mock_postprocessor,
        mock_segmentor,
        mock_pos_seed_selector,
        mock_neg_seed_selector,
        mock_graph_constructor,
        mock_patch_class,
        mock_embedding_class,
        mock_segmentation_class,
    ):
        mock_pos_seed_selector.select.return_value = "positive_seed"
        mock_neg_seed_selector.select.return_value = "negative_seed"
        mock_embedding_class.return_value = "embedding_class"
        mock_postprocessor.select.return_value = "best segmentation"
        mock_graph_constructor.construct.return_value = "graph"
        mock_patch_class.return_value = "patch"
        mock_segmentation_class.return_value.clean.return_value = "clean"

        center_seed = 1
        mock_segmentor.solve.return_value = [
            mock_segmentation_class(dummy, dummy),
            mock_segmentation_class(dummy, dummy),
        ]

        candidate = Candidate(center_seed, hnccorr)
        assert candidate.segment() == mock_postprocessor.select.return_value
        mock_pos_seed_selector.select.assert_called_once_with(center_seed, mock_movie)
        mock_neg_seed_selector.select.assert_called_once_with(center_seed, mock_movie)
        mock_patch_class.assert_called_once_with(mock_movie, center_seed, "patch_size")
        mock_embedding_class.assert_called_once_with(mock_patch_class.return_value)
        mock_graph_constructor.construct.assert_called_once_with(
            mock_patch_class.return_value, mock_embedding_class.return_value
        )
        mock_segmentor.solve.assert_called_once_with(
            mock_graph_constructor.construct.return_value,
            mock_pos_seed_selector.select.return_value,
            mock_neg_seed_selector.select.return_value,
        )
        mock_postprocessor.select.assert_called_once_with(["clean", "clean"])

    def test_candidate_equality(self):
        assert Candidate(1, "a") == Candidate(1, "a")
        assert Candidate(1, "a") != Candidate(2, "a")
        assert Candidate(1, "a") != Candidate(1, "b")

    def test_candidate_equality_wrong_class(self):
        class FakeCandidate:
            pass

        assert Candidate(1, "a") != FakeCandidate()

    def test_candidate_segmentations(self, hnccorr, mock_segmentor):
        c = Candidate(1, hnccorr)

        assert c.segmentations is None
        c.segment()
        assert c.segmentations == mock_segmentor.solve.return_value

    def test_candidate_center_seed(self):
        assert Candidate(1, "a").center_seed == 1

    def test_candidate_clean_segmentations(
        self, dummy, mock_segmentor, mock_segmentation_class, hnccorr
    ):
        mock_segmentor.solve.return_value = [
            mock_segmentation_class(dummy, dummy),
            mock_segmentation_class(dummy, dummy),
        ]
        mock_segmentation_class.return_value.clean.return_value = "clean"

        c = Candidate(1, hnccorr)
        assert c.clean_segmentations is None
        c.segment()
        assert c.clean_segmentations == ["clean", "clean"]

    def test_candidate_best_segmentations(self, hnccorr, mock_postprocessor):
        c = Candidate(1, hnccorr)
        assert c.best_segmentation is None
        c.segment()
        assert c.best_segmentation == mock_postprocessor.select.return_value


class TestHNCcorr:
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

    def test_segmentations_to_list(self, H, dummy, mock_segmentation_class):
        # selections are stored as lists to fix order.
        ms1 = mock_segmentation_class(dummy, dummy)
        ms1.selection = [(0, 1), (1, 0)]
        ms2 = deepcopy(ms1)
        ms2.selection = [(8, 3), (3, 10), (138, 23)]
        H.segmentations = [ms1, ms2]

        assert H.segmentations_to_list() == [
            {"coordinates": [(0, 1), (1, 0)]},
            {"coordinates": [(8, 3), (3, 10), (138, 23)]},
        ]


class TestHnccorrConfig:
    def test_config_attributes(self, config):
        assert config.patch_size == 31
        assert config.negative_seed_circle_radius == 10

    def test_invalid_parameter(self):
        with pytest.raises(ValueError):
            HNCcorrConfig(wrong_parameter=10)

    def test_config_add_wrong_class(self, config):
        class WrongClass:
            pass

        with pytest.raises(TypeError):
            config + WrongClass()

    def test_config_add_config(self, config):
        config2 = HNCcorrConfig(patch_size=21, positive_seed_radius=5)

        config3 = config + config2
        assert config.patch_size == 31
        assert config2.patch_size == 21
        assert config3.patch_size == 21

        with pytest.raises(AttributeError):
            config2.negative_seed_circle_radius
        assert config.negative_seed_circle_radius == 10
        assert config3.negative_seed_circle_radius == 10

        with pytest.raises(AttributeError):
            config.positive_seed_radius == 5
        assert config2.positive_seed_radius == 5
        assert config3.positive_seed_radius == 5

    def test_config_default_config(self):
        assert DEFAULT_CONFIG.seeder_mask_size == 3
        assert DEFAULT_CONFIG.seeder_exclusion_padding == 4
        assert DEFAULT_CONFIG.seeder_grid_size == 5
        assert DEFAULT_CONFIG.percentage_of_seeds == pytest.approx(0.4)
        assert DEFAULT_CONFIG.postprocessor_min_cell_size == 40
        assert DEFAULT_CONFIG.postprocessor_max_cell_size == 200
        assert DEFAULT_CONFIG.postprocessor_preferred_cell_size == 80
        assert DEFAULT_CONFIG.positive_seed_radius == 0
        assert DEFAULT_CONFIG.negative_seed_circle_radius == pytest.approx(10.0)
        assert DEFAULT_CONFIG.negative_seed_circle_count == 10
        assert DEFAULT_CONFIG.gaussian_similarity_alpha == pytest.approx(1.0)
        assert DEFAULT_CONFIG.sparse_computation_grid_distance == pytest.approx(
            1 / 35.0
        )
        assert DEFAULT_CONFIG.sparse_computation_dimension == 3
        assert DEFAULT_CONFIG.patch_size == 31
