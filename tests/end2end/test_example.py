# Copyright © 2017. Regents of the University of California (Regents). All Rights
# Reserved.
#
# Permission to use, copy, modify, and distribute this software and its documentation
# for educational, research, and not-for-profit purposes, without fee and without a
# signed licensing agreement, is hereby granted, provided that the above copyright
# notice, this paragraph and the following two paragraphs appear in all copies,
# modifications, and distributions. Contact The Office of Technology Licensing, UC
# Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201,
# for commercial licensing opportunities. Created by Quico Spaen, Roberto Asín-Achá,
# and Dorit S. Hochbaum, Department of Industrial Engineering and Operations Research,
# University of California, Berkeley.
#
# IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
# INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE
# OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE
# SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS
# IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES,
# ENHANCEMENTS, OR MODIFICATIONS.
from conftest import TEST_DATA_DIR
import numpy as np
import matplotlib.pyplot as plt

from hnccorr import HNCcorr, Movie, HNCcorrConfig
from hnccorr.example import load_example_data


def test_example():
    movie = Movie(
        "Example movie",
        load_example_data(
            filedir=TEST_DATA_DIR
        ),  # downloads sample Neurofinder dataset.
    )  # See documentation for alternatives
    H = HNCcorr.from_config(
        HNCcorrConfig(percentage_of_seeds=0.025)
    )  # Initialize HNCcorr with default configuration
    H.segment(movie)  # Identify cells in movie

    H.segmentations  # List of identified cells
    H.segmentations_to_list()  # Export list of cells (for Neurofinder)]

    A = np.zeros(movie.pixel_shape)
    for segmentation in H.segmentations:
        for i, j in segmentation.selection:
            A[i, j] += 1

    plt.figure(figsize=(6, 6))
    plt.imshow(A)
    plt.savefig("test cells neurofinder02.00.png")

    assert len(H.segmentations) >= 10
