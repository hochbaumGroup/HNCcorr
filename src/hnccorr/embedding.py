import numpy as np

from hnccorr.utils import add_time_index


class CorrelationEmbedding:
    def __init__(self, patch, alpha):
        data = patch[:].reshape(-1, np.product(patch.pixel_size))
        self.embedding = np.corrcoef(data.T).reshape(-1, *patch.pixel_size)
        self.embedding[np.isnan(self.embedding)] = 0
        self._length = self.embedding.shape[0]
        self._alpha = alpha

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


def exponential_distance_decay(self, first, second):
    return np.exp(-self._alpha * self.distance(first, second))
