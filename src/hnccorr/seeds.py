from hnccorr.utils import eight_neighborhood, add_offset_set_coordinates
from math import sin, cos, pi


class Seeds:
    def __init__(self, center_seed, pos_seeds, neg_seeds):
        self.center_seed = center_seed
        self.positive_seeds = pos_seeds
        self.negative_seeds = neg_seeds

    def select_positive_seeds(self, radius, movie_size):
        num_dimensions = len(movie_size)
        offsets = eight_neighborhood(num_dimensions, radius)
        # compute positive seeds
        positive_seeds = add_offset_set_coordinates(offsets, self.center_seed)
        # check if seeds are within boundaries
        return extract_valid_pixels(positive_seeds, movie_size)

    def select_negative_seeds(self, radius, count, movie_size):
        if len(movie_size) != 2:
            raise ValueError("Only 2-dimensional movies are currently supported.")

        angle_step = 2 * pi / float(count)

        angles = [i * angle_step for i in range(count)]
        offsets = {(round(radius * cos(x)), round(radius * sin(x))) for x in angles}
        negative_seeds = add_offset_set_coordinates(offsets, self.center_seed)

        # check if seeds are within boundaries
        return extract_valid_pixels(negative_seeds, movie_size)


def extract_valid_pixels(pixel_set, region_size):
    return {
        pixel
        for pixel in pixel_set
        if all([x >= 0 and x < dim_len for x, dim_len in zip(pixel, region_size)])
    }
