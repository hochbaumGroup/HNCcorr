from hnccorr.utils import eight_neighborhood, add_offset_set_coordinates


class Seeds:
    def __init__(self, center_seed, pos_seeds, neg_seeds):
        self.center_seed = center_seed
        self.positive_seeds = pos_seeds
        self.negative_seeds = neg_seeds

    def select_positive_seeds(self, radius, patch_size):
        num_dimensions = len(patch_size)
        offsets = eight_neighborhood(num_dimensions, radius - 1)
        # compute positive seeds
        positive_seeds = add_offset_set_coordinates(offsets, self.center_seed)
        # check if seeds are within boundaries
        positive_seeds = {
            seed
            for seed in positive_seeds
            if all(
                [
                    x >= 0 and x < dim_len
                    for x, dim_len in zip(seed, patch_size)
                ]
            )
        }
        return positive_seeds
