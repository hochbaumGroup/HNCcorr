from hnccorr.candidate import Candidate
from hnccorr.segmentation import Segmentation
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
    def from_config(cls, config):
        seeder = LocalCorrelationSeeder(3, 0.0005)
        postprocessor = SizePostprocessor(40, 200, 80)
        segmentor = HncParametric(0, 100000)
        positive_seed_selector = PositiveSeedSelector(0, [512, 512])
        negative_seed_selector = NegativeSeedSelector(10, 10, [512, 512])

        edge_selector = SparseComputation(3, 1 / 35.0)
        weight_function = lambda emb, a, b: exponential_distance_decay(emb, a, b, 1.0)
        graph_constructor = GraphConstructor(edge_selector, weight_function)
        patch_size = 31

        return cls(
            seeder,
            postprocessor,
            segmentor,
            positive_seed_selector,
            negative_seed_selector,
            graph_constructor,
            Candidate,
            Patch,
            CorrelationEmbedding,
            patch_size,
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
            segmentations = candidate.segment()
            self.segmentations.append(segmentations)
            seed = self.seeder.next()
        return self
