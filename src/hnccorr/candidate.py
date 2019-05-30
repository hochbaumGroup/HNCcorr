from hnccorr.segmentation import Segmentation


class Candidate:
    def __init__(self, value, postprocessor, segmentor):
        self._value = value
        self._postprocessor = postprocessor
        self._segmentor = segmentor
        self.segmentations = None
        self.best_segmentation = None

    def __eq__(self, other):
        return all(
            [
                self._value == other._value,
                self._postprocessor == other._postprocessor,
                self._segmentor == other._segmentor,
            ]
        )

    def segment(self):
        pos_seeds = None
        neg_seeds = None
        graph = None
        self.segmentations = self._segmentor.solve(graph, pos_seeds, neg_seeds)

        self.best_segmentation = self._postprocessor.select(self.segmentations)
        return self.best_segmentation
