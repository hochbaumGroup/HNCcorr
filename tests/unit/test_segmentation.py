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
from copy import copy
from itertools import product
import pytest
import networkx as nx


from hnccorr.segmentation import HncParametricWrapper, Segmentation


class TestHNC:
    def test_hnc_empty_so_nodes_in_sinkset(self):
        G = nx.DiGraph()
        G.add_nodes_from((i,) for i in range(7))

        h = HncParametricWrapper(0, 2)

        segmentations = h.solve(G, {(2,), (3,), (4,)}, {(1,), (5,)})

        assert len(segmentations) == 1
        assert segmentations[0].selection == {(2,), (3,), (4,)}
        assert segmentations[0].weight == pytest.approx(2.0)

    def test_hnc_edge_to_source_adjacent_node_results_in_node_selection(self):
        G = nx.DiGraph()
        G.add_nodes_from((i,) for i in range(7))
        G.add_edge((2,), (0,), weight=0.01)

        h = HncParametricWrapper(0, 2)

        segmentations = h.solve(G, {(2,), (3,), (4,)}, {(1,), (5,)})

        assert len(segmentations) == 1
        assert segmentations[0].selection == {(0,), (2,), (3,), (4,)}
        assert segmentations[0].weight == pytest.approx(2.0)


class TestSegmentation:
    def test_segmentation_weight(self):
        assert Segmentation({0, 1}, 0.5).weight == 0.5

    def test_segmentation_selection(self):
        assert Segmentation({(0, 1)}, 0.5).selection == {(0, 1)}

    def test_segmentation_equality_wrong_class(self):
        class FakeSegmentation:
            pass

        assert Segmentation({(0, 1)}, 0.5) != FakeSegmentation()

    def test_segmentation_clean_fill_hole(self):
        selection = {(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (1, 2), (2, 2)}
        weight = 1
        original = Segmentation(selection, weight)

        new = original.clean(set(), (5, 5))
        assert new == Segmentation(
            {(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)},
            weight,
        )
        assert original == Segmentation(selection, weight)

    def test_segmentation_clean_select_seed_component(self):
        selection = {(0, 0), (3, 3)}
        weight = 1
        original = Segmentation(selection, weight)

        new = original.clean({(3, 3)}, (5, 5))
        assert new == Segmentation({(3, 3)}, weight)
        assert original == Segmentation(selection, weight)
