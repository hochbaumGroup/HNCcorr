"""Base components of HNCcorr."""


from copy import deepcopy

from hnccorr.movie import Patch
from hnccorr.graph import (
    CorrelationEmbedding,
    exponential_distance_decay,
    GraphConstructor,
    SparseComputationEmbeddingWrapper,
)
from hnccorr.seeds import (
    PositiveSeedSelector,
    NegativeSeedSelector,
    LocalCorrelationSeeder,
)
from hnccorr.segmentation import HncParametricWrapper
from hnccorr.postprocessor import SizePostprocessor


class Candidate:
    """Encapsulates the logic for segmenting a single cell candidate / seed.

    Attributes:
        best_segmentation (Segmentation): Segmentation of a cell's spatial footprint as
            selected by the postprocessor.
        center_seed (tuple): Seed pixel coordinates.
        clean_segmentations (list[Segmentation]): List of segmentation after calling
            `clean()` on each segmentation.
        segmentations (list[Segmentation]): List of segmentations returned by HNC.
        _hnccorr (HNCcorr): HNCcorr object.
    """

    def __init__(self, center_seed, hnccorr):
        """Initialize Candidate object."""
        self.center_seed = center_seed
        self._hnccorr = hnccorr
        self.segmentations = None
        self.clean_segmentations = None
        self.best_segmentation = None

    def __eq__(self, other):
        """Compare Candidate object."""
        # pylint: disable=W0212
        if isinstance(other, Candidate):
            return (self.center_seed == other.center_seed) and (
                self._hnccorr == other._hnccorr
            )

        return False

    def segment(self):
        """Segment candidate cell and return footprint (if any).

        Encapsulates the procedure for segmenting a single cell candidate. It
        determines the seeds, constructs the similarity graph, and solves the HNC
        clustering problem for all values of the trade-off parameter lambda. The
        postprocessor selects the best segmentation or determines that no cell is found.

        Returns:
            Segmentation or None: Best segmentation or None if no cell is found.
        """
        movie = self._hnccorr.movie
        pos_seeds = self._hnccorr.positive_seed_selector.select(self.center_seed, movie)
        neg_seeds = self._hnccorr.negative_seed_selector.select(self.center_seed, movie)
        patch = self._hnccorr.patch_class(
            movie, self.center_seed, self._hnccorr.patch_size
        )
        embedding = self._hnccorr.embedding_class(patch)
        graph = self._hnccorr.graph_constructor.construct(patch, embedding)
        self.segmentations = self._hnccorr.segmentor.solve(graph, pos_seeds, neg_seeds)
        self.clean_segmentations = [
            s.clean(pos_seeds, movie.pixel_shape) for s in self.segmentations
        ]
        self.best_segmentation = self._hnccorr.postprocessor.select(self.segmentations)
        return self.best_segmentation


class HNCcorr:
    """Implementation of the HNCcorr algorithm.

    This class specifies all components of the algoritm and defines the procedure for
    segmenting the movie. How each candidate seed / location is evaluated is specified
    in the Candidate class.

    References:
        Q Spaen, R Asín-Achá, SN Chettih, M Minderer, C Harvey, and DS Hochbaum (2019).
        HNCcorr: A Novel Combinatorial Approach for Cell Identification in
        Calcium-Imaging Movies. eNeuro, 6(2).

    """

    def __init__(
        # pylint: disable=C0330
        self,
        seeder,
        postprocessor,
        segmentor,
        positive_seed_selector,
        negative_seed_selector,
        graph_constructor,
        candidate_class,
        patch_class,
        embedding_class,
        patch_size,
    ):
        """Initalizes HNCcorr object."""
        self.seeder = seeder
        self.postprocessor = postprocessor
        self.segmentor = segmentor
        self.positive_seed_selector = positive_seed_selector
        self.negative_seed_selector = negative_seed_selector
        self.graph_constructor = graph_constructor
        self._candidate_class = candidate_class
        self.patch_class = patch_class
        self.embedding_class = embedding_class
        self.patch_size = patch_size

        self.movie = None
        self.segmentations = []
        self.candidates = []

    @classmethod
    def from_config(cls, config=None):
        """Initializes HNCcorr from an HNCcorrConfig object.

        Provides a simple way to initialize an HNCcorr object from a configuration.
        Default components are used, and parameters are taken from the input
        configuration or inferred from the default configuration if not specified.

        Args:
            config (HNCcorrConfig): HNCcorrConfig object with modified configuration.
                Parameters that are not explicitly specified in the `config` object are
                taken from the default configuration ``DEFAULT_CONFIGURATION`` as
                defined in the `hnccorr.config` module.

        Returns:
            HNCcorr: Initialized HNCcorr object as parametrized by the configuration.
        """
        if config is None:
            config = DEFAULT_CONFIG
        else:
            config = DEFAULT_CONFIG + config

        edge_selector = SparseComputationEmbeddingWrapper(
            config.sparse_computation_dimension, config.sparse_computation_grid_distance
        )

        return cls(
            LocalCorrelationSeeder(
                config.seeder_mask_size,
                config.percentage_of_seeds,
                config.seeder_exclusion_padding,
            ),
            SizePostprocessor(
                config.postprocessor_min_cell_size,
                config.postprocessor_max_cell_size,
                config.postprocessor_preferred_cell_size,
            ),
            HncParametricWrapper(0, 1),
            PositiveSeedSelector(config.positive_seed_radius),
            NegativeSeedSelector(
                config.negative_seed_circle_radius, config.negative_seed_circle_count
            ),
            GraphConstructor(
                edge_selector,
                lambda a, b: exponential_distance_decay(
                    a, b, config.gaussian_similarity_alpha
                ),
            ),
            Candidate,
            Patch,
            CorrelationEmbedding,
            config.patch_size,
        )

    def segment(self, movie):
        """Applies the HNCcorr algorithm to identify cells in a calcium-imaging movie.

        Identifies cells the spatial footprints of cells in a calcium imaging movie.
        Cells are identified based on a set of candidate locations identified by the
        seeder. If a cell is found, the pixels in the spatial footprint are excluded as
        seeds for future segmentations. This prevents that a cell is segmented more
        than once. Although segmented pixels cannot seed a new segmentation, they may
        be segmented again.

        Identified cells are accessible through the `segmentations` attribute.

        Returns:
            Reference to itself.
        """
        self.movie = movie
        self.seeder.reset()
        self.segmentations = []
        self.candidates = []

        self.seeder.select_seeds(movie)

        seed = self.seeder.next()
        while seed is not None:
            candidate = self._candidate_class(seed, self)
            self.candidates.append(candidate)
            print(
                "Candidate: %d, Cells identified: %d"
                % (len(self.candidates), len(self.segmentations))
            )
            best_segmentation = candidate.segment()
            if best_segmentation is not None:
                self.segmentations.append(best_segmentation)
                self.seeder.exclude_pixels(best_segmentation.selection)
            seed = self.seeder.next()

        print("Completed - Total cells identified: %d" % len(self.segmentations))
        return self


class HNCcorrConfig:
    """Configuration class for HNCcorr algorithm.

    Enables tweaking the parameters of HNCcorr when used with the default components.
    Configurations are modular and can be combined using the addition operation.

    Each parameter is accessible as an attribute.

    Attributes:
        _entries (dict): Dict with parameter keys and values. Each parameter value
            (when defined) is also accessible as an attribute.
    """

    def __init__(self, **entries):
        """Initializes HNCcorrConfig object."""
        self._entries = entries

        for key, value in self._entries.items():
            setattr(self, key, value)

    def __add__(self, other):
        """Combines two configurations and returns a new one.

        If parameters are defined in both configurations, then `other` takes precedence.

        Args:
            other (HNCcorrConfig): Another configuration object.

        Returns:
            HNCcorrConfig: Configuration with combined parameter sets.

        Raises:
            TypeError: When other is not an instance of HNCcorrConfig.
        """
        if not isinstance(other, HNCcorrConfig):
            raise TypeError(
                "other is an instance of %s instead of %s." % (type(other), type(self))
            )

        entries = deepcopy(self._entries)
        entries.update(other._entries)  # pylint: disable=W0212

        return HNCcorrConfig(**entries)


DEFAULT_CONFIG = HNCcorrConfig(
    seeder_mask_size=3,
    seeder_exclusion_padding=4,
    percentage_of_seeds=0.40,
    postprocessor_min_cell_size=40,
    postprocessor_max_cell_size=200,
    postprocessor_preferred_cell_size=80,
    positive_seed_radius=0,
    negative_seed_circle_radius=10,
    negative_seed_circle_count=10,
    gaussian_similarity_alpha=1.0,
    sparse_computation_grid_distance=1 / 35.0,
    sparse_computation_dimension=3,
    patch_size=31,
)