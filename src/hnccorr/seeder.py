from itertools import product
import numpy as np

from hnccorr.utils import add_offset_set_coordinates, add_time_index
from hnccorr.patch import Patch


class LocalCorrelationSeeder(object):
    def __init__(
        self,
        movie,
        neighborhood_size=3,
        positive_seed_size=3,
        keep_fraction=0.4,
    ):
        self.neighborhood_size = neighborhood_size
        self.positive_seed_size = positive_seed_size
        self.keep_fraction = keep_fraction
        self._movie = movie

        self._select_seeds()

    def _select_seeds(self):
        # helpful constants
        max_shift = int((self.neighborhood_size - 1) / 2)

        # generate all offsets of neighbors
        neighbor_offsets = self._generate_offsets(max_shift)
        # remove point as neighbor
        neighbor_offsets = neighbor_offsets - {
            (0,) * self._movie.num_dimensions
        }

        mean_neighbor_corr = []

        for pixel in product(*[range(n) for n in self._movie.pixel_size]):
            pixel_data = self._movie[add_time_index(pixel)].reshape(1, -1)

            # compute neighbors
            neighbors = add_offset_set_coordinates(neighbor_offsets, pixel)

            # extract data for valid neighbors
            neighbors_data = []
            for neighbor in neighbors:
                if self._movie.is_valid_pixel_index(neighbor):
                    neighbors_data.append(
                        self._movie[add_time_index(neighbor)].reshape(1, -1)
                    )
            neighbors_data = np.concatenate(neighbors_data, axis=0)

            # compute correlation to each neighbor (corrcoef concatenates the
            # two vectors so we extract last row except for last element)
            neighbors_corr = np.corrcoef(neighbors_data, pixel_data)[-1, :-1]

            # store average correlation
            mean_neighbor_corr.append((pixel, np.mean(neighbors_corr)))

        mean_neighbor_corr = sorted(
            mean_neighbor_corr, key=lambda x: x[1], reverse=True
        )

        num_keep = int(self.keep_fraction * len(mean_neighbor_corr))

        # store best seeds
        self._seeds = [seed for seed, _ in mean_neighbor_corr[:num_keep]]
        self._current_index = 0

    def _generate_offsets(self, radius):
        return set(
            product(
                range(-radius, radius + 1), repeat=self._movie.num_dimensions
            )
        )

    def _construct_patch(self, seed):
        # compute offsets for neighboring points
        max_shift = int((self.positive_seed_size - 1) / 2)
        offsets = self._generate_offsets(max_shift)

        # compute positive seeds
        positive_seeds = add_offset_set_coordinates(offsets, seed)
        # check if seeds are within boundaries
        positive_seeds = {
            seed
            for seed in positive_seeds
            if self._movie.is_valid_pixel_index(seed)
        }
        return Patch(self._movie, seed, 7, positive_seeds)

    def next(self):
        if self._current_index < len(self._seeds):
            patch = self._construct_patch(self._seeds[self._current_index])
            self._current_index += 1

            return patch
        else:
            return None
