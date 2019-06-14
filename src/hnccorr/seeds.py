from math import sin, cos, pi

from hnccorr.utils import eight_neighborhood, add_offset_set_coordinates


class PositiveSeedSelector:
    """Selects positive seed pixels in a square centered on `center_seed`.

    Selects all pixels in a square centered on `center_seed` as positive seeds. A pixel is selected if it is within a Manhattan distance (L1) of `_max_distance` from the center seed pixel.

    Attributes:
        _max_distance (int): Maximum L1 distance allowed.
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


class NegativeSeedSelector:
    """Selects negative seed pixels uniformly from a circle around center seed pixel.

    Selects `_count` pixels from a circle centered on the center seed pixel with radius `_radius`. The selected pixels are spread uniformly over the circle. Non-integer
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
