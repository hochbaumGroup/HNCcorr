from hnccorr.candidate import Candidate
from hnccorr.segmentation import Segmentation


class HNCcorr:
    def __init__(self, seeder, postprocessor, segmentor):
        self.seeder = seeder
        self.segmentations = []
        self.postprocessor = postprocessor
        self.segmentor = segmentor
        self.candidates = []

    def segment(self, movie):
        self.seeder.reset()
        self.segmentations = []
        self.candidates = []

        self.seeder.select_seeds(movie)

        seed = self.seeder.next()
        while seed is not None:
            candidate = Candidate(seed, self)
            self.candidates.append(candidate)
            self.segmentations.append(Segmentation({(0, 1)}, 1.0))
            seed = self.seeder.next()
        return self
