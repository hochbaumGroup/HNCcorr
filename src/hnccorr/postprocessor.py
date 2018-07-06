import numpy as np


class SizePostprocessor(object):
    def __init__(self, min_size, max_size, pref_size):
        self._min_size = min_size
        self._max_size = max_size
        self._pref_size = pref_size

    def _filter(self, segmentations):
        return [
            s
            for s in segmentations
            if self._min_size <= len(s.selection) <= self._max_size
        ]

    def select(self, segmentations):
        """Select best candidate segmentation"""
        candidates = self._filter(segmentations)
        if not candidates:
            return None
        size = np.array([len(c.selection) for c in candidates])
        best_index = np.argmin(
            np.abs(np.sqrt(size) - np.sqrt(self._pref_size))
        )

        return candidates[best_index]
