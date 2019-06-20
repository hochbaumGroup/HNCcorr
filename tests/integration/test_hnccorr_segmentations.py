import pytest
import os
import numpy as np
import matplotlib.pyplot as plt

from conftest import TEST_DATA_DIR

from hnccorr.base import HNCcorr, HNCcorrConfig
from hnccorr.movie import Movie


@pytest.fixture
def data():
    return np.load(os.path.join(TEST_DATA_DIR, "neurofinder.02.00_10.npy"))


def test_hnccorr_single_segment(data):
    H = HNCcorr.from_config(HNCcorrConfig(percentage_of_seeds=0.0005))

    movie = Movie("Neurofinder02.00", data)

    H.segment(movie)

    A = np.zeros(movie.pixel_shape)
    for segmentation in H.segmentations:
        for i, j in segmentation.selection:
            A[i, j] += 1

    plt.figure(figsize=(6, 6))
    plt.imshow(A)
    plt.savefig("test cells neurofinder02.00.png")

    assert len(H.segmentations) >= 10
