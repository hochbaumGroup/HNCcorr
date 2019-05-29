from hnccorr.segmentation import Segmentation


class Candidate:
    def __init__(self, value):
        self._value = value
        self.segmentations = None

    def __eq__(self, other):
        return self._value == other._value

    def segment(self):
        self.segmentations = [Segmentation({(0, 1)}, 1.0)]

        best_segmentation = self.segmentations[0]
        return best_segmentation
