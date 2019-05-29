from hnccorr.segmentation import Segmentation


class Candidate:
    def __init__(self, value, postprocessor):
        self._value = value
        self._postprocessor = postprocessor
        self.segmentations = None
        self.best_segmentation = None

    def __eq__(self, other):
        return self._value == other._value

    def segment(self):
        self.segmentations = [Segmentation({(0, 1)}, 1.0)]

        self.best_segmentation = self.segmentations[0]
        return self.best_segmentation
