from hnccorr.candidate import Candidate
from hnccorr.segmentation import Segmentation


class HNCcorr:
    def __init__(self, seeder, postprocessor):
        self._seeder = seeder
        self.segmentations = []
        self.candidates = []

    def segment(self, movie):
        self._seeder.reset()
        self.segmentations = []
        self.candidates = []

        self._seeder.select_seeds(movie)

        seed = self._seeder.next()
        while seed is not None:
            candidate = Candidate(seed)
            self.candidates.append(candidate)
            self.segmentations.append(Segmentation({(0, 1)}, 1.0))
            seed = self._seeder.next()
        return self
