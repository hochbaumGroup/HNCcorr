from hnccorr.segmentation import Segmentation


class Candidate:
    def __init__(self, value, postprocessor, segmentor):
        self._value = value
        self._postprocessor = postprocessor
        self._segmentor = segmentor
        self.segmentations = None
        self.best_segmentation = None

    def __eq__(self, other):
        return (self._value == other._value) and (
            self._postprocessor == other._postprocessor
        )

    def segment(self):
        self.segmentations = self._segmentor.segment()

        self.best_segmentation = self._postprocessor.select(self.segmentations)
        return self.best_segmentation
