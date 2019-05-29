from hnccorr.segmentation import Segmentation


class Candidate:
    def __init__(self, value):
        self._value = value

    def __eq__(self, other):
        return self._value == other._value

    def segment(self):
        return Segmentation({(0, 1)}, 1.0)
