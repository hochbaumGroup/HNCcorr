import pytest
import os
import numpy as np

from conftest import TEST_DATA_DIR

from sparsecomputation import PCA
from hnccorr.movie import Movie, Patch
from hnccorr.hnccorr import HNCcorr
from hnccorr.candidate import Candidate
from hnccorr.embedding import CorrelationEmbedding, exponential_distance_decay
from hnccorr.graph import GraphConstructor
from hnccorr.seeds import (
    PositiveSeedSelector,
    NegativeSeedSelector,
    LocalCorrelationSeeder,
)
from hnccorr.edge_selection import SparseComputationEmbeddingWrapper
from hnccorr.postprocessor import SizePostprocessor
from hnccorr.candidate import Candidate
from hnccorr.segmentation import HncParametricWrapper, Segmentation


@pytest.fixture
def data():
    return np.load(os.path.join(TEST_DATA_DIR, "neurofinder.02.00_10.npy"))


@pytest.fixture
def matlab_segmentation():
    return Segmentation(
        {
            # First 3 pixels are manually ADDED due to small differences in
            # implementation.
            (285, 433),
            (282, 432),
            (279, 440),
            # Solution MATLAB:
            (280, 433),
            (281, 433),
            (282, 433),
            (283, 433),
            (284, 433),
            (279, 434),
            (280, 434),
            (281, 434),
            (282, 434),
            (283, 434),
            (284, 434),
            (285, 434),
            (286, 434),
            (279, 435),
            (280, 435),
            (281, 435),
            (282, 435),
            (283, 435),
            (284, 435),
            (285, 435),
            (286, 435),
            (287, 435),
            (279, 436),
            (280, 436),
            (281, 436),
            (282, 436),
            (283, 436),
            (284, 436),
            (285, 436),
            (286, 436),
            (287, 436),
            (279, 437),
            (280, 437),
            (281, 437),
            (282, 437),
            (283, 437),
            (284, 437),
            (285, 437),
            (286, 437),
            (287, 437),
            (279, 438),
            (280, 438),
            (281, 438),
            (282, 438),
            (283, 438),
            (284, 438),
            (285, 438),
            (286, 438),
            (287, 438),
            (279, 439),
            (280, 439),
            (281, 439),
            (282, 439),
            (283, 439),
            (284, 439),
            (285, 439),
            (286, 439),
            (280, 440),
            (281, 440),
            (282, 440),
            (283, 440),
            (284, 440),
            (285, 440),
            (286, 440),
            (280, 441),
            (281, 441),
            (282, 441),
            (283, 441),
            (284, 441),
            (285, 441),
            (282, 442),
            (283, 442),
        },
        0.0,
    )


def test_hnccorr_single_segment(mocker, dummy, data, matlab_segmentation):
    seeder = LocalCorrelationSeeder(3, 0.4, 4)
    postprocessor = SizePostprocessor(40, 200, 80)
    segmentor = HncParametricWrapper(0, 100000)
    positive_seed_selector = PositiveSeedSelector(0)

    # negative seed selector is mocked due to a bug in the matlab code.
    # Matlab passes the argument to sin(x)/cos(x) in degrees instead of radians.
    # This results in slightly wrong negative seeds.
    negative_seed_selector = mocker.patch(
        "hnccorr.seeds.NegativeSeedSelector", autospec=True
    )(dummy, dummy)
    negative_seed_selector.select.return_value = {
        (283, 447),
        (273, 435),
        (285, 427),
        (292, 440),
        (278, 445),
        (274, 431),
        (289, 429),
        (289, 444),
        (274, 442),
        (278, 427),
    }

    edge_selector = SparseComputationEmbeddingWrapper(
        3, 1 / 35.0, dimension_reducer=PCA(3)
    )
    graph_constructor = GraphConstructor(
        edge_selector, lambda a, b: exponential_distance_decay(a, b, 1.0)
    )
    patch_size = 31

    H = HNCcorr(
        seeder,
        postprocessor,
        segmentor,
        positive_seed_selector,
        negative_seed_selector,
        graph_constructor,
        Candidate,
        Patch,
        CorrelationEmbedding,
        patch_size,
    )

    H.movie = Movie("Neurofinder02.00", data)

    center_seed = (282, 436)
    c = Candidate(center_seed, H)
    best_segmentation = c.segment()

    assert best_segmentation.selection == matlab_segmentation.selection
    assert best_segmentation.weight == pytest.approx(matlab_segmentation.weight)
