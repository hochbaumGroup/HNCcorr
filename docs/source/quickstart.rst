Quickstart
============


Movies
--------

It all starts from a calcium-imaging movie. If your movie is stored as a numpy array, you can directly construct a :class:`~.Movie` object:

.. code-block:: python

    from hnccorr import Movie
    from hnccorr.example import load_example_data

    movie = Movie(
        "Example movie",  # Name of the movie
        load_example_data()  # Downloads sample Neurofinder dataset as a numpy array.
    )

If the movie is stored in tiff files, you can construct the :class:`~.Movie` object with :meth:`~.Movie.from_tiff_images`. This method loads a set of tiff files, each containing one frame, from a folder. The filenames should contain the frame numbers with zero-padding: 00001.tiff, 00002.tiff, 00003.tiff, etc. With  the ``memmap`` parameter you can specify whether the movie should be loaded into memory or a memory-mapped disk file should be created in the same folder. With the ``subsample``, you can specify how many frames should be subsampled into a single frame. By default, every 10 frames are averaged into a single frame.

.. caution::

    It is important that the tiff filenames are padded with zeros, such that they sort in the correct order.

Configuration
----------------

Before we construct the :class:`~.HNCcorr` object, we need to configure the algorithm with an :class:`~.HNCcorrConfig` object. The algorithm will perform better if some of the parameters are adjusted per dataset. For example, in the following example we adjust the minimum cell size for the postprocessor:

.. code-block:: python

    from hnccorr import HNCcorrConfig

    config = HNCcorrConfig(postprocessor_min_cell_size=80)

The default value is used for any parameter that is not explicitly specified in the configuration.

The adjustable parameters and their default values are:

* **postprocessor_min_cell_size** = 40: Lower bound on pixel count of a cell.
* **postprocessor_preferred_cell_size** = 80: Pixel count of a typical cell.
* **postprocessor_max_cell_size** = 200: Upper bound on pixel count of a cell.
* **patch_size** = 31: Size in pixel of each dimension of the patch.
* **positive_seed_radius** = 0: Radius of the positive seed square / superpixel.
* **negative_seed_circle_radius** = 10: Radius in pixels of the circle with negative seeds.
* **seeder_mask_size** = 3: Width in pixels of the region used by the seeder to compute the average correlation between a pixel and its neighbors.
* **seeder_grid_size (int)**: Size of grid bloc per dimension. Seeder maintains only the best candidate pixel for each grid block.
* **seeder_exclusion_padding** = 4: Distance for excluding additional pixels surrounding segmented cells.
* **percentage_of_seeds** = 0.40: Fraction of candidate seeds to evaluate.
* **negative_seed_circle_count** = 10: Number of negative seeds.
* **gaussian_similarity_alpha** = 1: Decay factor in gaussian similarity function.
* **sparse_computation_grid_distance** =  1 / 35.0 : 1 / grid_resolution. Width of each block in sparse computation.
* **sparse_computation_dimension** = 3: Dimension of the low-dimensional space in sparse computation.

The parameters at the top of the list are more likely to need adjust than those at the bottom of the list.

Cell identification
------------------------

Next, we construct the :class:`~.HNCcorr` object from its configuration:

.. code-block:: python

    H = HNCcorr.from_config(config)

Note that the ``config`` parameter is optional. If no configuration is specified, the default values for :class:`~.HNCcorr` are used.

We can then use :class:`~.HNCcorr` to segment the movie and extract the resulting segmentations:

.. code-block:: python

    H.segment(movie)

    H.segmentations  # List of identified cells
    H.segmentations_to_list()  # Export list of cells (for Neurofinder)
