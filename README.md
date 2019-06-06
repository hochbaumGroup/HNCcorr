# HNCcorr
The HNCcorr algorithm identifies cell bodies in two-photon calcium imaging movies.

HNCcorr is described in detail in our [eNeuro paper](http://www.eneuro.org/content/6/2/ENEURO.0304-18.2019):

Spaen, Q., Asín-Achá, R., Chettih, S. N., Minderer, M., Harvey, C., and Hochbaum, D. S. (2019). HNCcorr: A novel combinatorial approach for cell identification in calcium-imaging movies. eNeuro, 6(2).

We provide a Python 3 (recommended) implementation as well as a Matlab implementation. The software is freely available for non-commercial use (see license for details).

### Example (Python)
```python
from hnccorr import HNCcorr, Movie
from hnccorr.sample_data import sample_numpy_movie

movie = Movie("Sample movie", sample_numpy_movie)
H = HNCcorr.from_config() # Initialize HNCcorr with default configuration
H.segment(movie)


H.segmentations  # List of identified cells
H.segmentations_to_json_file()  # Export list of cells to JSON (for Neurofinder)
```

## Installation Instructions (Python 3)
You can install HNCcorr directly from the Python Package Index with pip:
```bash
pip install hnccorr
```
Or you can install your local copy by executing the following command from the main directory:
```bash
pip install .
```

On Windows you may need to install a [C-compiler for Python](https://wiki.python.org/moin/WindowsCompilers).

## Installation Instructions (Matlab)
The Matlab implementation was used to generate the results in the eNeuro manuscript and is now superseded by the Python implementation. The Matlab implementation is available in the `matlab` folder. See the README file in this folder for instructions.

## Documentation
The documentation is hosted at [ReadTheDocs](TBD).

## Hyperparameters
TBD

## Tests
The tests for HNCcorr use the `pytest` package. You can execute them with the `pytest` command in the main directory.
