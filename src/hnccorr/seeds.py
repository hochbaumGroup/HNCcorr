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
"""Seed related components of HNCcorr."""

from math import sin, cos, pi
import itertools
import numpy as np

from hnccorr.utils import (
    add_offset_set_coordinates,
    add_time_index,
    eight_neighborhood,
    generate_pixels,
)


class LocalCorrelationSeeder:
    """Provide seeds based on the correlation of pixels to their local neighborhood.

    Seed pixels are selected based on the average correlation of the pixel to its local
    neighborhood.For each block of `grid_size` by `grid_size` pixels, the pixel with
    the highest average local correlation is selected. The remaining pixels in each
    block are discarded. From the remaining pixels, a fraction of `_seed_fraction`
    pixels, those with the highest average local correlation, are kept and attempted
    for segmentation.

    The local neighborhood of each pixel consist of the pixels in a square of width
    `_neighborhood_size` centered on the pixels. Pixel coordinates outside the boundary
    of the movie are ignored.


    Attributes:
        _current_index (int): Index of next seed in `_seeds` to return.
        _excluded_pixels (set): Set of pixel coordinates to excluded as future seeds.
        _grid_size (int): Number of pixels per dimension in a block.
        _keep_fraction (float): Percentage of candidate seed pixels to attempt for
            segmentation. All other candidate seed pixels are discarded.
        _movie (Movie): Movie to segment.
        _neighborhood_size (int): Width in pixels of the local neighborhood of a pixel.
        _padding (int): L-infinity distance for determining which pixels should be
            padded to the exclusion set in `exclude_pixels()`.
        _seeds (list[tuple]): List of candidate seed coordinates to return.
    """

    def __init__(self, neighborhood_size, keep_fraction, padding, grid_size):
        """Initializes a LocalCorrelationSeeder object."""
        self._current_index = None
        self._excluded_pixels = set()
        self._keep_fraction = keep_fraction
        self._movie = None
        self._neighborhood_size = neighborhood_size
        self._padding = padding
        self._seeds = None
        self._grid_size = grid_size

    def select_seeds(self, movie):
        """Identifies candidate seeds in movie.

        Initializes list of candidate seeds in the movie. See class description for
        details. Seeds can be accessed via the :meth:`~.LocalCorrelationSeeder.next`
        method.

        Args:
            movie (Movie): Movie object to segment.

        Returns:
            None
        """
        self._movie = movie

        num_dimensions = self._movie.num_dimensions
        # helpful constants
        max_shift = int((self._neighborhood_size - 1) / 2)

        # generate all offsets of neighbors
        neighbor_offsets = eight_neighborhood(num_dimensions, max_shift)
        # remove point as neighbor
        neighbor_offsets = neighbor_offsets - {(0,) * num_dimensions}

        mean_neighbor_corr = {}

        for pixel in generate_pixels(self._movie.pixel_shape):
            # compute neighbors
            neighbors = add_offset_set_coordinates(neighbor_offsets, pixel)

            # extract data for valid neighbors
            valid_neighbors = [
                neighbor
                for neighbor in neighbors
                if self._movie.is_valid_pixel_coordinate(neighbor)
            ]

            # store average correlation
            mean_neighbor_corr[pixel] = self._compute_average_local_correlation(
                pixel, valid_neighbors
            )

        best_per_grid_block = self._select_best_per_grid_block(mean_neighbor_corr)

        best_per_grid_block_sorted = sorted(
            [(key, val) for key, val in best_per_grid_block.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        num_keep = int(self._keep_fraction * len(best_per_grid_block_sorted))

        # store best seeds
        self._seeds = [seed for seed, _ in best_per_grid_block_sorted[:num_keep]]
        self.reset()

    def _compute_average_local_correlation(self, pixel, valid_neighbors):
        """Compute average correlation between pixel and neighbors."""
        pixel_data = self._movie[add_time_index(pixel)].reshape(1, -1)

        # extract data for valid neighbors
        neighbors_data = []
        for neighbor in valid_neighbors:
            neighbors_data.append(self._movie[add_time_index(neighbor)].reshape(1, -1))
        neighbors_data = np.concatenate(neighbors_data, axis=0)

        # compute correlation to each neighbor (corrcoef concatenates the
        # two vectors so we extract last row except for last element)
        neighbors_corr = np.corrcoef(neighbors_data, pixel_data)[-1, :-1]

        # store average correlation
        return np.mean(neighbors_corr)

    def _select_best_per_grid_block(self, scores):
        """Selects pixel with highest score in a block of grid_size pixels per dim.

        """
        pixel_shape = self._movie.pixel_shape
        num_dimensions = self._movie.num_dimensions

        best_pixels = {}

        # identify all pixels in a block relatve to top left pixel as base coordinate
        shifts = set(
            itertools.product(*(range(self._grid_size) for _ in range(num_dimensions)))
        )

        # loop over all base coordinates of the grid, e.g. (0, 0), (5, 0), (0, 5),
        # (5, 5), (10, 0), ... for grid_size = 5
        for base_coordinate in itertools.product(
            *(range(0, pixel_shape[i], self._grid_size) for i in range(num_dimensions))
        ):
            # list of all coordinates in the block
            coordinates_in_block = add_offset_set_coordinates(shifts, base_coordinate)
            # select coordinate in block with highest score
            best_coordinate, best_value = max(
                [
                    (coordinate, scores[coordinate])
                    for coordinate in coordinates_in_block
                    if coordinate in scores
                ],
                key=lambda x: x[1],
            )

            best_pixels[best_coordinate] = best_value

        return best_pixels

    def exclude_pixels(self, pixels):
        """Excludes pixels from being returned by `next()` method.

        All pixels within in the set `pixels` as well as pixels that are within an L-
        infinity distance of `_padding` from any excluded pixel are excluded as seeds.

        Method enables exclusion of pixels in previously segmented cells from serving
        as new seeds. This may help to prevent repeated segmentation of the cell.

        Args:
            pixels (set): Set of pixel coordinates to exclude.

        Returns:
            None
        """
        neighborhood = eight_neighborhood(self._movie.num_dimensions, self._padding)

        padded_pixel_sets = [
            add_offset_set_coordinates(neighborhood, pixel) for pixel in pixels
        ]

        self._excluded_pixels = self._excluded_pixels.union(
            pixels.union(*padded_pixel_sets)
        )

    def next(self):
        """Provides next seed pixel for segmentation.

        Returns the movie coordinates of the next available seed pixel for
        segmentation. Seed pixels that have previously been excluded will be ignored.
        Returns None when all seeds are exhausted.

        Returns:
            tuple or None: Coordinates of next seed pixel. None if no seeds remaining.
        """
        while self._current_index < len(self._seeds):
            center_seed = self._seeds[self._current_index]
            self._current_index += 1

            if center_seed not in self._excluded_pixels:
                return center_seed

        return None

    def reset(self):
        """Reinitialize the sequence of seed pixels and empties `_excluded_seeds`."""
        self._current_index = 0
        self._excluded_pixels = set()


class NegativeSeedSelector:
    """Selects negative seed pixels uniformly from a circle around center seed pixel.

    Selects `_count` pixels from a circle centered on the center seed pixel with radius
    `_radius`. The selected pixels are spread uniformly over the circle. Non-integer
    pixel indices are rounded to the closest (integer) pixel. Currently only
    2-dimensional movies are supported.

    Attributes:
        _radius (float): L2 distance to center seed.
        _count (int): Number of negative seed pixels to select.
    """

    def __init__(self, radius, count):
        self._radius = radius
        self._count = count

    def select(self, center_seed, movie):
        """Selects negative seed pixels.

        Args:
            center_seed (tuple): Center seed pixels.
            movie (Movie): Movie for segmentation.

        Returns:
            set: Set of negative seed pixels. Each pixel is denoted by a tuple.
        """
        if movie.num_dimensions != 2:
            raise ValueError("Only 2-dimensional movies are currently supported.")

        # Determine uniform locations of pixels on the circle in radians.
        angle_step = 2 * pi / float(self._count)
        angles = [i * angle_step for i in range(self._count)]
        offsets = {
            (round(self._radius * cos(x)), round(self._radius * sin(x))) for x in angles
        }
        negative_seeds = add_offset_set_coordinates(offsets, center_seed)

        # check if seeds are within boundaries
        return movie.extract_valid_pixels(negative_seeds)


class PositiveSeedSelector:
    """Selects positive seed pixels in a square centered on `center_seed`.

    Selects all pixels in a square centered on `center_seed` as positive seeds. A pixel
    is selected if it is within a Chebyshev distance (L-Inf) of `_max_distance` from the
    center seed pixel.

    Attributes:
        _max_distance (int): Maximum L-Inf distance allowed.
    """

    def __init__(self, max_distance):
        self._max_distance = max_distance

    def select(self, center_seed, movie):
        """Selects positive seeds.

        Args:
            center_seed (tuple): Center seed pixel.
            movie (Movie): Movie for segmentation.

        Returns:
            set: Set of positive seed pixels. Each pixel is denoted by a tuple.
        """
        offsets = eight_neighborhood(movie.num_dimensions, self._max_distance)
        # compute positive seeds
        positive_seeds = add_offset_set_coordinates(offsets, center_seed)
        # check if seeds are within boundaries
        return movie.extract_valid_pixels(positive_seeds)
