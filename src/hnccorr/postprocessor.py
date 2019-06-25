# Copyright © 2017. Regents of the University of California (Regents). All Rights
# Reserved.
#
# Permission to use, copy, modify, and distribute this software and its documentation
# for educational, research, and not-for-profit purposes, without fee and without a
# signed licensing agreement, is hereby granted, provided that the above copyright
# notice, this paragraph and the following two paragraphs appear in all copies,
# modifications, and distributions. Contact The Office of Technology Licensing, UC
# Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201,
# for commercial licensing opportunities. Created by Quico Spaen, Roberto Asín-Achá,
# and Dorit S. Hochbaum, Department of Industrial Engineering and Operations Research,
# University of California, Berkeley.
#
# IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
# INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE
# OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE
# SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS
# IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES,
# ENHANCEMENTS, OR MODIFICATIONS.
"""Postprocesser component for selecting the best segmentation in HNCcorr."""

import numpy as np


class SizePostprocessor:
    """Selects the best segmentation based on the number of selected pixels.

    Discards all segmentations that contain more pixels than ``_max_size`` or less
    pixels then ``_min_size``. If no segmentations remains, no cell was found and
    ``None`` is returned. Otherwise the segmentation is returned that minimizes
    ``|sqrt(x) - sqrt(_pref_size)|`` where x is the number of pixels in the
    segmentation.

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
