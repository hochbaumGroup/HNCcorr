import numpy as np


def test_embedding(P):
    from hnccorr.embeddings import CorrelationEmbedding

    np.testing.assert_allclose(
        CorrelationEmbedding(P((0,))).embedding[0],
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
