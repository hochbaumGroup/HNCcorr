import pytest
import numpy as np

from hnccorr.embedding import CorrelationEmbedding, exponential_distance_decay


@pytest.fixture
def CE1(P):
    return CorrelationEmbedding(P((0,)))


@pytest.fixture
def CE2(P2):
    return CorrelationEmbedding(P2)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_embedding(CE1, CE2):
    np.testing.assert_allclose(
        CE1.embedding[0],
        [1.0, 0.99833749, 0.99339927, 0.98532928, 0.9743547, 0.96076892, 0.94491118],
    )

    np.testing.assert_allclose(CE2.embedding[(0, 0, slice(None, None))], [0, 0, 0])


def test_correlation_embedding(CE1):
    CE1.embedding = np.array([[0.0, 1.0], [-2.0, 0.0]])
    assert CE1.distance((0,), (1,)) == pytest.approx(2.5)
    assert CE1.distance((0,), (0,)) == pytest.approx(0)


def test_exponential_distance_decay(CE1):
    alpha = 0.5
    exponential_distance_decay(CE1, (0,), (1,), alpha) == pytest.approx(
        np.exp(-0.003866 * alpha)
    )
