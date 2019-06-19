from hnccorr.config import DEFAULT_CONFIG
from hnccorr.candidate import Candidate
from hnccorr.patch import Patch
from hnccorr.embedding import CorrelationEmbedding, exponential_distance_decay
from hnccorr.graph import GraphConstructor
from hnccorr.seeds import (
    PositiveSeedSelector,
    NegativeSeedSelector,
    LocalCorrelationSeeder,
)
from hnccorr.edge_selection import SparseComputationEmbeddingWrapper
from hnccorr.hnc import HncParametricWrapper
from hnccorr.postprocessor import SizePostprocessor


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
