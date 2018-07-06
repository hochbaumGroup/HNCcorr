from itertools import product
import numpy as np

from hnccorr.utils import (
    add_offset_coordinates,
    add_offset_set_coordinates,
    eight_neighborhood,
)


class Patch(object):
    """Subregion of movie with seeds to evaluate for cell presence
    """

    def __init__(
        self,
        movie,
        center_seed,
        window_size,
        negative_seed_radius,
        positive_seeds,
    ):
        if window_size % 2 == 0:
            raise ValueError("window_size (%d) should be an odd number.")

        self._num_dimensions = movie.num_dimensions
        self._window_size = window_size
        self._negative_seed_radius = negative_seed_radius
        self._movie = movie
        self.pixel_size = (window_size,) * self._num_dimensions
        self.num_frames = movie.num_frames
        self._center_seed = center_seed

        self.coordinate_offset = self._compute_coordinate_offset()

        offset = list(-x for x in self.coordinate_offset)
        self.positive_seeds = add_offset_set_coordinates(
            positive_seeds, offset
        )
        self.negative_seeds = self._select_negative_seeds()

        self._data = self._movie[self._movie_indices()]

    def __eq__(self, other):
        return (
            self._movie == other._movie
            and self._center_seed == other._center_seed
            and self.pixel_size == other.pixel_size
            and self._negative_seed_radius == other._negative_seed_radius
            and self.positive_seeds == other.positive_seeds
            and self.negative_seeds == other.negative_seeds
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
        half_width = int((self._window_size - 1) / 2)

        topleft_coordinates = add_offset_coordinates(
            self._center_seed, (-half_width,) * self._num_dimensions
        )
        # shift left such that top left corner exists
        topleft_coordinates = list(max(x, 0) for x in topleft_coordinates)

        # bottomright corners (python-style index so not included)
        bottomright_coordinates = add_offset_coordinates(
            topleft_coordinates, (self._window_size,) * self._num_dimensions
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
            (-self._window_size,) * self._num_dimensions,
        )

        return topleft_coordinates

    def _movie_indices(self):
        """Compute movie index range of patch"""
        bottomright_coordinates = add_offset_coordinates(
            self.coordinate_offset, (self._window_size,) * self._num_dimensions
        )

        idx = [slice(None, None)]
        for start, stop in zip(
            self.coordinate_offset, bottomright_coordinates
        ):
            idx.append(slice(start, stop))
        return idx

    def __getitem__(self, key):
        return self._data[key]


class PatchFactory(object):
    def __init__(self, movie, window_size, negative_seed_radius):
        self._movie = movie
        self._window_size = window_size
        self._negative_seed_radius = negative_seed_radius

    def construct(self, center_seed, positive_seeds):
        return Patch(
            self._movie,
            center_seed,
            self._window_size,
            self._negative_seed_radius,
            positive_seeds,
        )
