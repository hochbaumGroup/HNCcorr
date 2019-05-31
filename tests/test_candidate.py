import pytest

from hnccorr.hnccorr import HNCcorr
from hnccorr.candidate import Candidate


@pytest.fixture
def mock_pos_seed_selector(mocker, dummy):
    pos_seed_selector = mocker.patch(
        "hnccorr.seeds.PositiveSeedSelector", autospec=True
    )(dummy, dummy)
    pos_seed_selector.select.return_value = "positive_seed"
    return pos_seed_selector


@pytest.fixture
def mock_neg_seed_selector(mocker, dummy):
    neg_seed_selector = mocker.patch(
        "hnccorr.seeds.NegativeSeedSelector", autospec=True
    )(dummy, dummy, dummy)
    neg_seed_selector.select.return_value = "negative_seed"
    return neg_seed_selector


@pytest.fixture
def mock_segmentor(mocker, dummy):
    segmentor = mocker.patch("hnccorr.hnc.HncParametric", autospec=True)(dummy, dummy)
    segmentor.solve.return_value = "segmentations"
    return segmentor


@pytest.fixture
def mock_postprocessor(mocker, dummy):
    postprocessor = mocker.patch(
        "hnccorr.postprocessor.SizePostprocessor", autospec=True
    )(dummy, dummy, dummy)
    postprocessor.select.return_value = "best segmentation"
    return postprocessor


@pytest.fixture
def mock_graph_constructor(mocker, dummy):
    graph_constructor = mocker.patch("hnccorr.graph.GraphConstructor", autospec=True)(
        dummy, dummy
    )
    graph_constructor.construct.return_value = "graph"
    return graph_constructor


@pytest.fixture
def mock_patch_class(mocker, dummy):
    patch_class = mocker.patch("hnccorr.patch.Patch", autospec=True)
    patch_class.return_value = "patch"
    return patch_class


@pytest.fixture
def mock_embedding_class(mocker, dummy):
    embedding_class = mocker.patch(
        "hnccorr.embedding.CorrelationEmbedding", autospec=True
    )
    embedding_class.return_value = "embedding_class"
    return embedding_class


@pytest.fixture
def hnccorr(
    dummy,
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
        mock_patch_class,
        mock_embedding_class,
    )
    H.movie = "movie"
    return H


def test_candidate_segment(
    hnccorr,
    mock_postprocessor,
    mock_segmentor,
    mock_pos_seed_selector,
    mock_neg_seed_selector,
    mock_graph_constructor,
    mock_patch_class,
    mock_embedding_class,
):
    center_seed = 1

    assert (
        Candidate(center_seed, hnccorr).segment()
        == mock_postprocessor.select.return_value
    )
    mock_pos_seed_selector.select.assert_called_once_with(center_seed)
    mock_neg_seed_selector.select.assert_called_once_with(center_seed)
    mock_patch_class.assert_called_once_with("movie", center_seed, "patch_size")
    mock_embedding_class.assert_called_once_with(mock_patch_class.return_value, "alpha")
    mock_graph_constructor.construct.assert_called_once_with(
        mock_patch_class.return_value, mock_embedding_class.return_value
    )
    mock_segmentor.solve.assert_called_once_with(
        mock_graph_constructor.construct.return_value,
        mock_pos_seed_selector.select.return_value,
        mock_neg_seed_selector.select.return_value,
    )
    mock_postprocessor.select.assert_called_once_with(mock_segmentor.solve.return_value)


def test_candidate_equality():
    assert Candidate(1, "a") == Candidate(1, "a")
    assert Candidate(1, "a") != Candidate(2, "a")
    assert Candidate(1, "a") != Candidate(1, "b")


def test_candidate_segmentations(hnccorr, mock_segmentor):
    c = Candidate(1, hnccorr)
    assert c.segmentations is None
    c.segment()
    assert c.segmentations == mock_segmentor.solve.return_value


def test_candidate_best_segmentations(hnccorr, mock_postprocessor):
    c = Candidate(1, hnccorr)
    assert c.best_segmentation is None
    c.segment()
    assert c.best_segmentation == mock_postprocessor.select.return_value
