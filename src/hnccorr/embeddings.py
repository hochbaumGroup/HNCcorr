import numpy as np

from hnccorr.utils import add_time_index


class CorrelationEmbedding(object):
    def __init__(self, P):
        data = P[:]
        self.embedding = np.corrcoef(data.T)
        self._length = self.embedding.shape[0]

    def distance(self, first, second):
        return (
            np.sqrt(
                np.sum(
                    np.power(
                        self.embedding[add_time_index(first)]
                        - self.embedding[add_time_index(second)],
                        2,
                    )
                )
            )
            / self._length
        )
