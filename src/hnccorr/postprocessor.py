import numpy as np


class SizePostprocessor:
    """Selects the best segmentation based on the number of selected pixels.

    Discards all segmentations that contain more pixels than ``_max_size`` or less
    pixels then ``_min_size``. If no segmentations remains, no cell was found and
    ``None`` is returned. Otherwise the segmentation is returned that minimizes
    ``|sqrt(x) - sqrt(_pref_size)|`` where x is the number of pixels in the segmentation.

    Attributes:
        _min_size (int): Lower bound for the cell size in pixels.
        _max_size (int): Upper bound for the cell size in pixels.
        _pref_size (int): Preferred cell size in pixels.
    """

    def __init__(self, min_size, max_size, pref_size):
        """Initializes a SizePostprocessor object."""
        self._min_size = min_size
        self._max_size = max_size
        self._pref_size = pref_size

    def _filter(self, segmentations):
        """Returns a list of segmentations with size between min_size and max_size."""
        return [
            s
            for s in segmentations
            if self._min_size <= len(s.selection) <= self._max_size
        ]

    def select(self, segmentations):
        """Selects the best segmentation based on the number of selected pixels.

        See class description for details.

        Args:
            segmentations (List[Segmentation]): List of candidate segmentations.

        Returns:
            Segmentation or None: Best segmentation or None if all are discarded.
        """
        candidates = self._filter(segmentations)
        if not candidates:
            return None
        size = np.array([len(c.selection) for c in candidates])
        best_index = np.argmin(np.abs(np.sqrt(size) - np.sqrt(self._pref_size)))

        return candidates[best_index]
