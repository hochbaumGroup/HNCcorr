def test_example():
    from hnccorr import HNCcorr, Movie
    from hnccorr.example import example_numpy_data

    movie = Movie(
        "Example movie", example_numpy_data
    )  # See documentation for alternatives
    H = HNCcorr.from_config()  # Initialize HNCcorr with default configuration
    H.segment(movie)  # Identify cells in movie

    H.segmentations  # List of identified cells
    H.segmentations_to_list()  # Export list of cells (for Neurofinder)
