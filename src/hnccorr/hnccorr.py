from hnccorr.config import DEFAULT_CONFIG
from hnccorr.candidate import Candidate
from hnccorr.patch import Patch
from hnccorr.embedding import CorrelationEmbedding, exponential_distance_decay
from hnccorr.graph import GraphConstructor
from hnccorr.seeds import PositiveSeedSelector, NegativeSeedSelector
from hnccorr.edge_selection import SparseComputation
from hnccorr.hnc import HncParametric
from hnccorr.seeder import LocalCorrelationSeeder
from hnccorr.postprocessor import SizePostprocessor


class HNCcorr:
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
        if config is None:
            config = DEFAULT_CONFIG
        else:
            config = DEFAULT_CONFIG + config

        edge_selector = SparseComputation(
            config.sparse_computation_dimension, config.sparse_computation_grid_distance
        )
        weight_function = lambda emb, a, b: exponential_distance_decay(
            emb, a, b, config.gaussian_similarity_alpha
        )

        return cls(
            LocalCorrelationSeeder(config.seeder_mask_size, config.percentage_of_seeds),
            SizePostprocessor(
                config.postprocessor_min_cell_size,
                config.postprocessor_max_cell_size,
                config.postprocessor_preferred_cell_size,
            ),
            HncParametric(0, 1),
            PositiveSeedSelector(config.positive_seed_radius),
            NegativeSeedSelector(
                config.negative_seed_circle_radius,
                config.negative_seed_circle_count,
                [512, 512],
            ),
            GraphConstructor(edge_selector, weight_function),
            Candidate,
            Patch,
            CorrelationEmbedding,
            config.patch_size,
        )

    def segment(self, movie):
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
