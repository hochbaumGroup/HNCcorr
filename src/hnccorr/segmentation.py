from hnccorr.utils import fill_holes, select_max_seed_component


class Segmentation(object):
    def __init__(self, selection, weight):
        self.selection = set(selection)
        self.weight = weight

    def __eq__(self, other):
        return True

    def clean(self, positive_seeds, region_size):
        """Remove left over points / fill holes"""
        self.selection = select_max_seed_component(
            self.selection, positive_seeds, len(region_size)
        )
        self.selection = fill_holes(self.selection, region_size)
        return self
