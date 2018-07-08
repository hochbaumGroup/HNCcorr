import numpy as np


class CorrelationEmbedding(object):
    def __init__(self, P):
        data = P[:]
        self.embedding = np.corrcoef(data.T)
