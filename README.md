# HNCcorr

[![Current version](https://img.shields.io/pypi/v/hnccorr.svg)](https://pypi.python.org/pypi/hnccorr)
[![Documentation](https://readthedocs.org/projects/hnccorr/badge/?version=latest&style=flat)](https://hnccorr.readthedocs.io)
![Travis-CI status](https://travis-ci.com/hochbaumGroup/HNCcorr.svg?branch=master)

The HNCcorr algorithm identifies cell bodies in two-photon calcium imaging movies. We provide a Python 3 implementation as well as a legacy Matlab implementation. The software is freely available for non-commercial use. See license for details.

The HNCcorr algorithm is described in our [eNeuro paper](http://www.eneuro.org/content/6/2/ENEURO.0304-18.2019):

> Q Spaen, R Asín-Achá, SN Chettih, M Minderer, C Harvey, and DS Hochbaum. (2019). HNCcorr: A novel combinatorial approach for cell identification in calcium-imaging movies. eNeuro, 6(2).

### Example (Python)
```python
from hnccorr import HNCcorr, Movie
from hnccorr.example import load_example_data

movie = Movie(
    "Example movie", load_example_data()  # downloads sample Neurofinder dataset
)
H = HNCcorr.from_config()  # Initialize HNCcorr with default configuration
H.segment(movie)

H.segmentations  # List of identified cells
H.segmentations_to_list()  # Export list of cells (for Neurofinder)
```

See the [quickstart](https://hnccorr.readthedocs.io/en/latest/quickstart.html) guide for more details.

## Installation Instructions for Python 3
You can install HNCcorr directly from the Python Package Index with pip:
```bash
pip install hnccorr
```

On Windows you may need to install a [C-compiler for Python](https://wiki.python.org/moin/WindowsCompilers).

## Installation Instructions for Matlab
The Matlab implementation was used to generate the results in the eNeuro manuscript and is now superseded by the Python implementation. The Matlab implementation is available in the `matlab` folder. See the README file in the `matlab` folder for instructions.

## Documentation
The documentation is hosted at [ReadTheDocs](https://hnccorr.readthedocs.io).

## Tests
The tests for HNCcorr use the `pytest` package. You can execute them with the `pytest` command in the main directory.
