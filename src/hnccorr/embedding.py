import numpy as np

from hnccorr.utils import add_time_index


class CorrelationEmbedding:
    """Represents each pixel as a vector of correlations to other pixels."""

    def __init__(self, patch):
        data = patch[:].reshape(-1, np.product(patch.pixel_shape))
        self.embedding = np.corrcoef(data.T).reshape(-1, *patch.pixel_shape)
        self.embedding[np.isnan(self.embedding)] = 0
        self._length = self.embedding.shape[0]

    def get_vector(self, pixel):
        return self.embedding[add_time_index(pixel)]


def exponential_distance_decay(feature_vec1, feature_vec2, alpha):
    return np.exp(-alpha * np.mean(np.power(feature_vec1 - feature_vec2, 2)))
