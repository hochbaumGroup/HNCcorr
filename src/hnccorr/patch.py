from hnccorr.utils import (
    add_offset_coordinates,
    add_time_index,
    generate_pixels,
    add_offset_set_coordinates,
)


class Patch:
    """Square subregion of Movie.

    Patch limits the data used for the segmentation of a potential cell. Given a center
    seed pixel, Patch defines a square subregion centered on the seed pixel with width
    patch_size. If the square extends outside the movie boundaries, then the subregion
    is shifted such that it stays within the movie boundaries.

    The patch also provides an alternative coordinate system with respect to the top
    left pixel of the patch. This pixel is the zero coordinate for the patch coordinate
    system. The coordinate offset is the coordinate of the top left pixel in the movie
    coordinate system.

    Attributes:
        _center_seed (tuple): Seed pixel that marks the potential cell. The pixel is
            represented as a tuple of coordinates. The coordinates are relative to the
            movie. The top left pixel of the movie represents zero.
        _coordinate_offset (tuple): Movie coordinates of the pixel that represents the
            zero coordinate in the Patch object. Similar to the Movie, pixels in the
            Patch are indexed from the top left corner.
        _data (np.array): Subset of the Movie data. Only data for the patch is stored.
        _movie (Movie): Movie for which the Patch object is a subregion.
        _num_dimensions (int): Dimension of the patch. It matches the dimension of the
            movie.
        _patch_size (int): length of the patch in each dimension. Must be an odd number.
    """

    def __init__(self, movie, center_seed, patch_size):
        """Initializes Patch object."""
        if patch_size % 2 == 0:
            raise ValueError("patch_size (%d) should be an odd number.")

        self._num_dimensions = movie.num_dimensions
        self._center_seed = center_seed
        self._patch_size = patch_size
        self._movie = movie
        self._coordinate_offset = self._compute_coordinate_offset()
        self._data = self._movie[self._movie_indices()]

    @property
    def num_frames(self):
        """Number of frames in the Movie."""
        return self._movie.num_frames

    @property
    def pixel_shape(self):
        """Shape of the patch in pixels. Does not not included the time dimension."""
        return (self._patch_size,) * self._num_dimensions

    def _compute_coordinate_offset(self):
        """Computes the coordinate offset of the patch.

        Confirms that the patch falls within the movie boundaries and shifts the patch
        if necessary. The center seed pixel may not be in the center of the patch if a
        shift is necessary.
        """
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
            for x, max_value in zip(bottomright_coordinates, self._movie.pixel_shape)
        )

        topleft_coordinates = add_offset_coordinates(
            bottomright_coordinates, (-self._patch_size,) * self._num_dimensions
        )

        return topleft_coordinates

    def _movie_indices(self):
        """Computes the indices of the movie that correspond to the patch.

        For a patch with top left pixel (5, 5) and bottom right pixel (9, 9), this
        method returns ``(:, 5:10, 5:10)`` which can be used to acccess the data
        corresponding to the patch in the movie.
        """
        bottomright_coordinates = add_offset_coordinates(
            self._coordinate_offset, (self._patch_size,) * self._num_dimensions
        )

        # pixel indices
        idx = []
        for start, stop in zip(self._coordinate_offset, bottomright_coordinates):
            idx.append(slice(start, stop))
        return add_time_index(tuple(idx))

    def to_movie_coordinate(self, patch_coordinate):
        """Converts a movie coordinate into a patch coordinate.

        Args:
            patch_coordinate (tuple): Coordinates of a pixel in patch coordinate system.

        Returns:
            tuple: Coordinate of pixel in movie coordinate system.
        """
        return add_offset_coordinates(patch_coordinate, self._coordinate_offset)

    def to_patch_coordinate(self, movie_coordinate):
        """Converts a movie coordinate into a patch coordinate.

        Args:
            movie_coordinate (tuple): Coordinates of a pixel in movie coordinate system.

        Returns:
            tuple: Coordinate of pixel in patch coordinate system.
        """
        return add_offset_coordinates(
            movie_coordinate, [-x for x in self._coordinate_offset]
        )

    def enumerate_pixels(self):
        """Returns the movie coordinates of the pixels in the patch."""
        return add_offset_set_coordinates(
            generate_pixels(self.pixel_shape), self._coordinate_offset
        )

    def __getitem__(self, key):
        """Access data for pixels in the patch. Indexed in patch coordinates."""
        return self._data[key]
