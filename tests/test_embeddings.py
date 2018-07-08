import pytest
import numpy as np


@pytest.fixture
def CE(P):
    from hnccorr.embeddings import CorrelationEmbedding

    return CorrelationEmbedding(P((0,)))


def test_embedding(CE):
    np.testing.assert_allclose(
        CE.embedding[0],
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


def test_correlation_embedding(CE):
    CE.distance((0,), (1,)) == pytest.approx(0.003866)
