from hnccorr.candidate import Candidate


class HNCcorr:
    def __init__(self, seeder):
        self._seeder = seeder

    def segment(self, movie):
        self.candidates = []
        self._seeder.select_seeds(movie)

        seed = self._seeder.next()
        while seed is not None:
            candidate = Candidate(seed)
            self.candidates.append(candidate)
            seed = self._seeder.next()
        return self
