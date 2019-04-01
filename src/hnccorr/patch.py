import numpy as np

from hnccorr.segmentation import Segmentation
from hnccorr.utils import (
    add_offset_coordinates,
    add_offset_set_coordinates,
    eight_neighborhood,
    add_time_index,
)
from hnccorr.hnc import HNC
from hnccorr.seeds import Seeds
from hnccorr.graph import GraphConstructor
from hnccorr.embedding import CorrelationEmbedding, exponential_distance_decay
from hnccorr.edge_selection import SparseComputation
from hnccorr.utils import eight_neighborhood


class Patch(object):
    """Subregion of movie with seeds to evaluate for cell presence
    """

    def __init__(self, movie, config, center_seed, patch_size):
        self._num_dimensions = movie.num_dimensions
        self._center_seed = center_seed
        self._config = config
        self._patch_size = patch_size
        self._movie = movie
        self.pixel_size = (patch_size,) * self._num_dimensions
        self.num_frames = movie.num_frames

        if patch_size % 2 == 0:
            raise ValueError("patch_size (%d) should be an odd number.")

        self.coordinate_offset = self._compute_coordinate_offset()

        offset = list(-x for x in self.coordinate_offset)

        self._data = self._movie[self._movie_indices()]

    def _compute_coordinate_offset(self):
        half_width = int((self._patch_size - 1) / 2)

        topleft_coordinates = add_offset_coordinates(
            self._center_seed, (-half_width,) * self._num_dimensions
        )
        # shift left such that top left corner exists
        topleft_coordinates = list(max(x, 0) for x in topleft_coordinates)

        # bottomright corners (python-style index so not included)
        bottomright_coordinates = add_offset_coordinates(
            topleft_coordinates, (self._patch_size,) * self._num_dimensions
        )
        # shift right such that bottom right corner exists
        bottomright_coordinates = list(
            min(x, max_value)
            for x, max_value in zip(
                bottomright_coordinates, self._movie.pixel_size
            )
        )

        topleft_coordinates = add_offset_coordinates(
            bottomright_coordinates,
            (-self._patch_size,) * self._num_dimensions,
        )

        return topleft_coordinates

    def _movie_indices(self):
        """Compute movie index range of patch"""
        bottomright_coordinates = add_offset_coordinates(
            self.coordinate_offset, (self._patch_size,) * self._num_dimensions
        )

        idx = []
        for start, stop in zip(
            self.coordinate_offset, bottomright_coordinates
        ):
            idx.append(slice(start, stop))
        return add_time_index(tuple(idx))

    def __getitem__(self, key):
        return self._data[key]
