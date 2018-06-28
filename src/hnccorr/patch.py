from hnccorr.utils import add_offset_coordinates, add_offset_set_coordinates


class Patch(object):
    """Subregion of movie with seeds to evaluate for cell presence
    """

    def __init__(self, movie, center_seed, window_size, positive_seeds):
        if window_size % 2 == 0:
            raise ValueError("window_size (%d) should be an odd number.")

        self._num_dimensions = movie.num_dimensions
        self._window_size = window_size
        self._movie = movie
        self.pixel_size = (window_size,) * self._num_dimensions
        self.num_frames = movie.num_frames
        self._center_seed = center_seed

        self.coordinate_offset = self._compute_coordinate_offset()

        offset = list(-x for x in self.coordinate_offset)
        self.positive_seeds = add_offset_set_coordinates(
            positive_seeds, offset
        )

        self._data = self._movie[self._movie_indices()]

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
            self.coordinate_offset,
            (self._window_size,) * self._num_dimensions,
        )

        idx = [slice(None, None)]
        for start, stop in zip(
            self.coordinate_offset, bottomright_coordinates
        ):
            idx.append(slice(start, stop))
        return idx

    def __getitem__(self, key):
        return self._data[key]
