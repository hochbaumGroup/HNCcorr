class Candidate:
    """Encapsulates the logic for segmenting a single cell candidate / seed.

    Attributes:
        best_segmentation (Segmentation): Segmentation of a cell's spatial footprint as
            selected by the postprocessor.
        center_seed (tuple): Seed pixel coordinates.
        clean_segmentations (list[Segmentation]): List of segmentation after calling
            `clean()` on each segmentation.
        segmentations (list[Segmentation]): List of segmentations returned by HNC.
        _hnccorr (HNCcorr): HNCcorr object.
    """

    def __init__(self, center_seed, hnccorr):
        """Initialize Candidate object."""
        self.center_seed = center_seed
        self._hnccorr = hnccorr
        self.segmentations = None
        self.clean_segmentations = None
        self.best_segmentation = None

    def __eq__(self, other):
        """Compare Candidate object."""
        # pylint: disable=W0212
        if isinstance(other, Candidate):
            return (self.center_seed == other.center_seed) and (
                self._hnccorr == other._hnccorr
            )

        return False

    def segment(self):
        """Segment candidate cell and return footprint (if any).

        Encapsulates the procedure for segmenting a single cell candidate. It
        determines the seeds, constructs the similarity graph, and solves the HNC
        clustering problem for all values of the trade-off parameter lambda. The
        postprocessor selects the best segmentation or determines that no cell is found.

        Returns:
            Segmentation or None: Best segmentation or None if no cell is found.
        """
        movie = self._hnccorr.movie
        pos_seeds = self._hnccorr.positive_seed_selector.select(self.center_seed, movie)
        neg_seeds = self._hnccorr.negative_seed_selector.select(self.center_seed, movie)
        patch = self._hnccorr.patch_class(
            movie, self.center_seed, self._hnccorr.patch_size
        )
        embedding = self._hnccorr.embedding_class(patch)
        graph = self._hnccorr.graph_constructor.construct(patch, embedding)
        self.segmentations = self._hnccorr.segmentor.solve(graph, pos_seeds, neg_seeds)
        self.clean_segmentations = [
            s.clean(pos_seeds, movie.pixel_shape) for s in self.segmentations
        ]
        self.best_segmentation = self._hnccorr.postprocessor.select(self.segmentations)
        return self.best_segmentation
