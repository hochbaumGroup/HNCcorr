from hnccorr.utils import add_offset_coordinates, add_offset_set_coordinates


class Patch(object):
    """Subregion of movie with seeds to evaluate for cell presence
    """
    def __init__(self, movie, center_seed, window_size, positive_seeds):
        if window_size % 2 == 0:
            raise ValueError("window_size (%d) should be an odd number.")
        else:
            half_width = int((window_size - 1) / 2)

        self._num_dimensions = len(movie.pixel_size)
        self._window_size = window_size
        self._movie = movie
        self.pixel_size = (window_size, ) * self._num_dimensions
        self.num_frames = movie.num_frames

        self._center_seed = center_seed
        self._topleft_coordinates = add_offset_coordinates(
            center_seed, (-half_width,) * self._num_dimensions
        )

        offset = list(-x for x in self._topleft_coordinates)
        self.positive_seeds = add_offset_set_coordinates(
            positive_seeds, offset
        )

        self._data = self._movie[self._movie_indices()]

    def _movie_indices(self):
        """Compute movie index range of patch"""
        bottomright_coordinates = add_offset_coordinates(
            self._topleft_coordinates,
            (self._window_size,) * self._num_dimensions,
        )

        idx = [slice(None, None)]
        for start, stop in zip(
            self._topleft_coordinates, bottomright_coordinates
        ):
            idx.append(slice(start, stop))
        return idx

    def __getitem__(self, key):
        return self._data[key]
