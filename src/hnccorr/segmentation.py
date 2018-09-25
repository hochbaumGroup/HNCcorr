from hnccorr.utils import fill_holes, select_max_seed_component


class Segmentation(object):
    def __init__(self, patch, selection, weight):
        self._patch = patch
        self.selection = set(selection)
        self.weight = weight

    def __eq__(self, other):
        return (
            self._patch == other._patch
            and self.selection == other.selection
            and self.weight == other.weight
        )

    def clean(self):
        """Remove left over points / fill holes"""
        self.selection = select_max_seed_component(
            self.selection,
            self._patch.seeds.positive_seeds,
            len(self._patch.pixel_size),
        )
        self.selection = fill_holes(self.selection, self._patch.pixel_size)
