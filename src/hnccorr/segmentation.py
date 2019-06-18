from itertools import product
import networkx as nx
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes

from hnccorr.utils import four_neighborhood


class Segmentation:
    """A set of pixels identified by HNC as a potential cell footprint.

    Attributes:
        selection (set): Pixels in the spatial footprint. Each pixel is represented as
            a tuple.
        weight (float): Upper bound on the lambda coefficient for which this
            segmentation is optimal.
    """

    def __init__(self, selection, weight):
        """Initializes a Segmentation object."""
        self.selection = set(selection)
        self.weight = weight

    def __eq__(self, other):
        """Compares two Segmentation objects."""
        if isinstance(other, Segmentation):
            return (self.selection == other.selection) and (self.weight == other.weight)

        return False

    def clean(self, positive_seeds, movie_pixel_shape):
        """Cleans Segmentation by selecting a connected component and filling holes.

        The Segmentation is decomposed into connected components by considering
        horizontal or vertical adjacent pixels as neighbors. The connected component
        with the most positive seeds is selected. Any holes in the selected component
        are added to the selection.

        Args:
            positive_seeds (set): Pixels that are contained in the
                spatial footprint. Each pixel is represented by a tuple.
            movie_pixel_shape (tuple): Pixel resolution of the movie.

        Returns:
            Segmentation: A new Segmentation object with the same weight.
        """
        improved_segmentation = self.select_max_seed_component(positive_seeds)
        return improved_segmentation.fill_holes(movie_pixel_shape)

    def select_max_seed_component(self, positive_seeds):
        """Selects the connected component of selection that contains the most seeds.

        The Segmentation is decomposed into connected components by considering
        horizontal or vertical adjacent pixels as neighbors. The connected component
        with the most positive seeds is selected.

        Args:
            positive_seeds (set): Pixels that are contained in the
                spatial footprint. Each pixel is represented by a tuple.

        Returns:
            Segmentation: A new Segmentation object with the same weight.
        """

        # get an arbitrary element from seeds to compute dimension
        num_dims = len(next(iter(self.selection)))
        neighbors = four_neighborhood(num_dims)

        graph = nx.Graph()
        graph.add_nodes_from(self.selection)

        for index, shift in product(self.selection, neighbors):
            neighbor = tuple(map(lambda a, b: a + b, index, shift))
            if neighbor in graph:
                graph.add_edge(index, neighbor)

        components = list(nx.connected_components(graph))

        overlap = [len(c.intersection(positive_seeds)) for c in components]

        best_component = components[np.argmax(overlap)]

        return Segmentation(best_component, self.weight)

    def fill_holes(self, movie_pixel_shape):
        """Fills holes in the selection.

        Args:
            movie_pixel_shape (tuple): Pixel resolution of the movie.

        Returns:
            Segmentation: A new Segmentation object with the same weight.
        """
        mask = np.full(movie_pixel_shape, False, dtype=np.bool)

        indices = list(zip(*self.selection))
        mask[indices] = True

        filled_mask = binary_fill_holes(mask)

        index_arrays = [a.tolist() for a in np.where(filled_mask)]
        return Segmentation(set(zip(*index_arrays)), self.weight)
