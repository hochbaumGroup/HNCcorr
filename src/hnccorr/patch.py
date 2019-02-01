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

    def __init__(
        self, movie, config, center_seed, patch_size, negative_seed_radius
    ):
        self._num_dimensions = movie.num_dimensions
        self._center_seed = center_seed
        self._config = config
        self._patch_size = patch_size
        self._negative_seed_radius = negative_seed_radius
        self._positive_seed_size = 3
        self._movie = movie
        self.pixel_size = (patch_size,) * self._num_dimensions
        self.num_frames = movie.num_frames

        if patch_size % 2 == 0:
            raise ValueError("patch_size (%d) should be an odd number.")

        self.coordinate_offset = self._compute_coordinate_offset()

        offset = list(-x for x in self.coordinate_offset)

        positive_seeds = self._select_positive_seeds()

        positive_seeds = add_offset_set_coordinates(positive_seeds, offset)
        negative_seeds = self._select_negative_seeds()
        self.seeds = Seeds(center_seed, positive_seeds, negative_seeds)

        self._data = self._movie[self._movie_indices()]

    def _select_positive_seeds(self):
        # compute offsets for neighboring points
        max_shift = int((self._positive_seed_size - 1) / 2)
        offsets = eight_neighborhood(self._num_dimensions, max_shift)
        # compute positive seeds
        positive_seeds = add_offset_set_coordinates(offsets, self._center_seed)
        # check if seeds are within boundaries
        positive_seeds = {
            seed
            for seed in positive_seeds
            if self._movie.is_valid_pixel_index(seed)
        }
        return positive_seeds

    def __eq__(self, other):
        return (
            self._movie == other._movie
            and self.seeds.center_seed == other.seeds.center_seed
            and self.pixel_size == other.pixel_size
            and self._negative_seed_radius == other._negative_seed_radius
            and self.seeds.positive_seeds == other.seeds.positive_seeds
            and self.seeds.negative_seeds == other.seeds.negative_seeds
            and self.coordinate_offset == other.coordinate_offset
        )

    def _select_negative_seeds(self):
        dist = np.zeros(
            (2 * self._negative_seed_radius + 1,) * self._num_dimensions
        )

        for coordinates in eight_neighborhood(
            self._num_dimensions, self._negative_seed_radius
        ):
            coordinates = np.array(coordinates)
            index = coordinates + self._negative_seed_radius
            dist[index] = np.linalg.norm(coordinates)

        indices_negative_seeds = np.where(
            np.floor(dist) == self._negative_seed_radius
        )

        indices_list = [x.tolist() for x in indices_negative_seeds]
        # convert indices to coordinates
        negative_seeds = set(
            tuple(i - self._negative_seed_radius for i in x)
            for x in zip(*indices_list)
        )
        # shift to center seed location
        negative_seeds = add_offset_set_coordinates(
            negative_seeds, self._center_seed
        )

        # remove seeds outside of movie boundary
        negative_seeds = {
            x for x in negative_seeds if self._movie.is_valid_pixel_index(x)
        }

        return add_offset_set_coordinates(
            negative_seeds, [-x for x in self.coordinate_offset]
        )

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

    def segment(self):
        embedding = CorrelationEmbedding(self, 0.5)

        graph_constructor = GraphConstructor(
            self,
            SparseComputation(3, 1 / 25.0),
            lambda a, b: exponential_distance_decay(embedding, 0, a, b),
        )
        graph = graph_constructor.construct(embedding)
        hnc = HNC(self, graph, graph_constructor.arc_weight)
        return hnc.solve_parametric(0, 2)
