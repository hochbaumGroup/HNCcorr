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
"""Provides example data for HNCcorr."""
import errno
import os
import numpy as np
from tqdm import tqdm
from six.moves import urllib


def load_example_data(  # pylint: disable=C0330
    filedir=".", filename="neurofinder.02.00_agg10.npy", download=True, memmap=False
):
    """Downloads a subsampled copy of the Neurofinder 02.00 dataset.

    Dataset is subsampled in the time dimension, where every 10 frames are replaced by
    a single frame with average intensity values. Resulting data has 800 frames, each
    of 512 x 512 pixels.

    Args:
        filedir (str): Directory for data file. Default is current directory.
        filename (str): Filename for data file.
        download (bool): If True and file does not exist, a new copy is downloaded. If
            False, data array is only loaded.
        memmap (bool): If True, a array-like memory map for the data is returned. If
            False, the data is loaded into memory. Default is False.

    Returns:
        np.array: Returns Numpy array like copy of the movie data.
    """
    url = "https://hnccorr-example-data.s3-us-west-2.amazonaws.com/neurofinder.02.00_agg10.npy"  # pylint: disable=C0301

    if memmap:
        memmap_mode = "r"
    else:
        memmap_mode = None

    filepath = os.path.join(filedir, filename)

    if os.path.isfile(filepath):
        pass
    elif download:
        download_url(url, filedir, filename=filename)
    else:
        raise IOError(
            "The file %s in directory %s does not exist. Set download=True."
            % (filename, os.path.expanduser(filedir))
        )

    return np.load(filepath, mmap_mode=memmap_mode)


def gen_bar_updater():
    """Provides an progress meter. Taken from Pytorch. See licence in source code."""
    # BSD 3-Clause License
    #
    # Copyright (c) Soumith Chintala 2016,
    # All rights reserved.
    #
    # Redistribution and use in source and binary forms, with or without modification,
    # are permitted provided that the following conditions are met:
    #
    # * Redistributions of source code must retain the above copyright notice, this
    # list of conditions and the following disclaimer.
    #
    # * Redistributions in binary form must reproduce the above copyright notice, this
    # list of conditions and the following disclaimer in the documentation and/or other
    # materials provided with the distribution.
    #
    # * Neither the name of the copyright holder nor the names of its contributors may
    # be used to endorse or promote products derived from this software without
    # specific prior written permission.
    #
    # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    # ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    # WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
    # ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    # (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
    # OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    # THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    # NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
    # IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    pbar = tqdm(total=None, unit="MB", mininterval=0.5)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size / (1024.0 ** 2)
        progress_bytes = count * block_size / (1024.0 ** 2)
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def download_url(url, root, filename=None):
    """Download a file from a url and place it in root.

    Taken from Pytorch. See licence in source code.

    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the
            basename of the URL
    """
    # BSD 3-Clause License
    #
    # Copyright (c) Soumith Chintala 2016,
    # All rights reserved.
    #
    # Redistribution and use in source and binary forms, with or without modification,
    # are permitted provided that the following conditions are met:
    #
    # * Redistributions of source code must retain the above copyright notice, this
    # list of conditions and the following disclaimer.
    #
    # * Redistributions in binary form must reproduce the above copyright notice, this
    # list of conditions and the following disclaimer in the documentation and/or other
    # materials provided with the distribution.
    #
    # * Neither the name of the copyright holder nor the names of its contributors may
    # be used to endorse or promote products derived from this software without
    # specific prior written permission.
    #
    # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    # ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    # WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
    # ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    # (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
    # OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    # THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    # NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
    # IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    # check if path exists.
    try:
        os.makedirs(root)
    except OSError as error:
        if error.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    try:
        print("Downloading " + url + " to " + fpath)
        urllib.request.urlretrieve(url, fpath, reporthook=gen_bar_updater())
    except urllib.error.URLError as error:
        if url[:5] == "https":
            url = url.replace("https:", "http:")
            print(
                "Failed download. Trying https -> http instead."
                " Downloading " + url + " to " + fpath
            )
            urllib.request.urlretrieve(url, fpath, reporthook=gen_bar_updater())
        else:
            raise error
