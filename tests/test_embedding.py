import pytest
import numpy as np

from hnccorr.embedding import CorrelationEmbedding
from hnccorr.embedding import exponential_distance_decay


@pytest.fixture
def CE1(P):
    return CorrelationEmbedding(P((0,)), 0.5)


@pytest.fixture
def CE2(P2):
    return CorrelationEmbedding(P2, 0.5)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_embedding(CE1, CE2):
    np.testing.assert_allclose(
        CE1.embedding[0],
        [
            1.0,
            0.99833749,
            0.99339927,
            0.98532928,
            0.9743547,
            0.96076892,
            0.94491118,
        ],
    )

    np.testing.assert_allclose(
        CE2.embedding[(0, 0, slice(None, None))], [0, 0, 0]
    )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_correlation_embedding(CE1, CE2):
    CE1.distance((0,), (1,)) == pytest.approx(0.003866)
    np.seterr(divide="ignore")
    CE2.distance((0,), (1,)) == pytest.approx(0.0)


def test_exponential_distance_decay(CE1):
    exponential_distance_decay(CE1, (0,), (1,)) == pytest.approx(
        np.exp(-0.003866 * 0.5)
    )
