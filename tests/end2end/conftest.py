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
import pytest
import os
import numpy as np

from hnccorr.example import load_example_data

TEST_DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../test_data"
)


@pytest.fixture
def dummy():
    return "Dummy"


@pytest.fixture
def neurofinder_data():
    return load_example_data(filedir=TEST_DATA_DIR)
