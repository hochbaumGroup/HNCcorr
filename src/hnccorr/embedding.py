import numpy as np

from hnccorr.utils import add_time_index


class CorrelationEmbedding:
    def __init__(self, patch):
        data = patch[:].reshape(-1, np.product(patch.pixel_shape))
        self.embedding = np.corrcoef(data.T).reshape(-1, *patch.pixel_shape)
        self.embedding[np.isnan(self.embedding)] = 0
        self._length = self.embedding.shape[0]

    def distance(self, first, second):
        return np.mean(
            np.power(
                self.embedding[add_time_index(first)]
                - self.embedding[add_time_index(second)],
                2,
            )
        )


def exponential_distance_decay(embedding, first, second, alpha):
    return np.exp(-alpha * embedding.distance(first, second))
