API Documentation
==================
Here you can find the details of the various HNCcorr components.

This HNCcorr implementation has the following components:

* **Candidate** - Contains the logic for segmenting a single cell.
* **Embedding** - Provides the feature vector of each pixel.
* **GraphConstructor** - Constructs the similarity graph.
* **HNC** - Solves Hochbaum's Normalized Cut (HNC) on a given similarity graph.
* **HNCcorr** - Provides the overal logic for segmenting all cells in a movie.
* **Movie** - Provides access to the data of a calcium imaging movie.
* **Patch** - Represents a square subregion of a movie (used for segmenting a cell).
* **Positive / negative seed selector** -- Selects positive or negative seed pixels in a patch.
* **Post-processor** - Selects the best segmentation (if any) for a cell.
* **Seeder** - Generates candidate cell locations.
* **Segmentation** - Represents a candidate segmentation of a cell.

Submodules
----------

.. toctree::

   hnccorr.base
   hnccorr.graph
   hnccorr.movie
   hnccorr.postprocessor
   hnccorr.seeds
   hnccorr.segmentation
   hnccorr.utils

Module contents
---------------

.. autoclass:: hnccorr.base.HNCcorr
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:

.. autoclass:: hnccorr.base.HNCcorrConfig
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:

.. autoclass:: hnccorr.movie.Movie
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:
