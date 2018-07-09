import pytest
import numpy as np


@pytest.fixture
def CE():
    from hnccorr.embedding import CorrelationEmbedding

    return lambda x: CorrelationEmbedding(x)


@pytest.fixture
def CE1(CE, P):
    return CE(P((0,)))


@pytest.fixture
def CE2(CE, P2):
    return CE(P2)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_embedding(CE1, CE2):
    np.testing.assert_allclose(
        CE1.embedding[0],
        [
            1.,
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
    CE2.distance((0,), (1,)) == pytest.approx(0.)


def test_exponential_distance_decay(CE1):
    from hnccorr.embedding import exponential_distance_decay

    exponential_distance_decay(CE1, 0.5, (0,), (1,)) == pytest.approx(
        np.exp(-0.003866 * 0.5)
    )
