from copy import deepcopy


class HNCcorrConfig:
    def __init__(self, **entries):
        self._entries = entries

        for key, value in self._entries.items():
            setattr(self, key, value)

    def __add__(self, other):
        entries = deepcopy(self._entries)
        entries.update(other._entries)  # pylint: disable=W0212

        return HNCcorrConfig(**entries)


DEFAULT_CONFIG = HNCcorrConfig(
    seeder_mask_size=3,
    percentage_of_seeds=0.40,
    postprocessor_min_cell_size=40,
    postprocessor_max_cell_size=200,
    postprocessor_preferred_cell_size=80,
    positive_seed_radius=0,
    negative_seed_circle_radius=10,
    negative_seed_circle_count=10,
    gaussian_similarity_alpha=1.0,
    sparse_computation_grid_distance=1 / 35.0,
    sparse_computation_dimension=3,
    patch_size=31,
)
