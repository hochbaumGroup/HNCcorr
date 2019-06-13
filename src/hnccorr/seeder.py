import numpy as np

from hnccorr.utils import (
    add_offset_set_coordinates,
    add_time_index,
    eight_neighborhood,
    generate_pixels,
)


class LocalCorrelationSeeder:
    def __init__(self, neighborhood_size, keep_fraction, padding):
        self._neighborhood_size = neighborhood_size
        self._keep_fraction = keep_fraction
        self._movie = None
        self._num_dims = None
        self._padding = padding
        self._seeds = None
        self._current_index = None
        self._excluded_pixels = set()

    def select_seeds(self, movie):
        self._movie = movie
        self._num_dims = self._movie.num_dimensions
        # helpful constants
        max_shift = int((self._neighborhood_size - 1) / 2)

        # generate all offsets of neighbors
        neighbor_offsets = eight_neighborhood(self._num_dims, max_shift)
        # remove point as neighbor
        neighbor_offsets = neighbor_offsets - {(0,) * self._num_dims}

        mean_neighbor_corr = []

        for pixel in generate_pixels(self._movie.pixel_size):
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

        num_keep = int(self._keep_fraction * len(mean_neighbor_corr))

        # store best seeds
        self._seeds = [seed for seed, _ in mean_neighbor_corr[:num_keep]]
        self._current_index = 0

    def exclude_pixels(self, pixels):
        neighborhood = eight_neighborhood(self._num_dims, self._padding)

        padded_pixel_sets = [
            add_offset_set_coordinates(neighborhood, pixel) for pixel in pixels
        ]

        self._excluded_pixels = self._excluded_pixels.union(
            pixels.union(*padded_pixel_sets)
        )

    def next(self):
        while self._current_index < len(self._seeds):
            center_seed = self._seeds[self._current_index]
            self._current_index += 1

            if center_seed not in self._excluded_pixels:
                return center_seed

        return None

    def reset(self):
        self._current_index = 0
