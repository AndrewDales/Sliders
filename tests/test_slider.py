import copy

import pytest
from numpy import array

from slider import Slider, SliderNode, find_a_star_path, find_full_route, create_partition
import slider


@pytest.fixture
def slider_node():
    return [SliderNode(3, 4, [2, 1, 10, 9, 5, 7, 3, 8, 6, 11, 4, 0], move_weight=0.5),
            SliderNode(3, 2, [5, 0, 2, 1, 4, 3], target_positions={0, 3}, move_weight=1),
            SliderNode(3, 3, [6, 5, 4, 3, 8, 2, 1, 7, 0], move_weight=1),
            SliderNode(4, 4, [14, 2, 6, 10, 15, 12, 4, 8, 11, 5, 13, 7, 3, 9, 1, 0], move_weight=0.5)]


class TestSlider:
    @pytest.fixture
    def valid_slider(self):
        return {'s_2_3': Slider(2, 3, [0, 1, 5, 3, 4, 2]),
                's_4_4': Slider(4, 4, [9, 4, 0, 6, 12, 13, 14, 3, 2, 7, 5, 15, 10, 1, 8, 11])}

    def test_default_set_up(self):
        s_root = Slider()
        assert s_root.data == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        assert s_root.solution == (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
        assert s_root.n_rows == 4
        assert s_root.n_cols == 4
        assert s_root.blank_pos == 15
        assert isinstance(s_root, Slider)

    def test__find_row_col(self):
        assert (Slider._find_row_col(7, 3) == array((2, 1))).all()
        assert (Slider._find_row_col(12, 4) == array((3, 0))).all()

    def test__find_pos_index(self):
        assert Slider._find_pos_index(array((2, 3)), 4) == 11

    def test_is_soluble(self, valid_slider):
        assert valid_slider['s_2_3'].is_soluble
        assert valid_slider['s_4_4'].is_soluble
        assert not Slider(2, 3, [0, 1, 2, 4, 3, 5]).is_soluble

    def test__swap(self, valid_slider):
        valid_slider['s_4_4']._swap(0, 1)
        assert valid_slider['s_4_4'].data == [4, 9, 0, 6, 12, 13, 14, 3, 2, 7, 5, 15, 10, 1, 8, 11]
        assert not valid_slider['s_4_4'].is_soluble
        valid_slider['s_2_3']._swap(0, 2)
        assert valid_slider['s_2_3'].blank_pos == 0
        assert not valid_slider['s_2_3'].is_soluble
        valid_slider['s_2_3']._swap(1, 2)
        assert valid_slider['s_2_3'].is_soluble

    def test_find_blank_neighbours(self, valid_slider):
        n_1 = valid_slider['s_2_3'].find_blank_neighbours()
        assert set(n_1) == {"R", "U"}
        assert (n_1["R"] == array((0, 1))).all()
        assert (n_1["U"] == array((1, 2))).all()
        n_2 = valid_slider['s_4_4'].find_blank_neighbours(fixed_cells={12, 13, 14, 15})
        assert set(n_2) == {"D", "R"}

    def test_shuffle(self, valid_slider):
        s = valid_slider['s_2_3']
        for _ in range(10):
            s.shuffle()
            assert s.is_soluble
        assert sorted(s.data) == sorted(s.solution)
        assert s.blank_pos == s.data.index(max(s.data))

    def test_slide(self, valid_slider):
        valid_slider['s_2_3'].slide("R")
        assert valid_slider['s_2_3'].is_soluble
        assert valid_slider['s_2_3'].data == [0, 5, 1, 3, 4, 2]
        valid_slider['s_4_4'].slide("U")
        assert valid_slider['s_4_4'].data == [9, 4, 0, 6, 12, 13, 14, 3, 2, 7, 5, 11, 10, 1, 8, 15]

    def test_dist_from_solution(self, valid_slider):
        assert valid_slider['s_2_3'].dist_from_solution() == 2
        # Check distance measure when only the first 4 values are included
        assert valid_slider['s_4_4'].dist_from_solution((0, 1, 2, 3)) == 10


class TestSliderNode:
    @pytest.fixture(autouse=True)
    def _create_node(self, slider_node):
        self.nodes = slider_node

    def test_create(self):
        assert self.nodes[0].n_rows == 3
        assert self.nodes[0].n_cols == 4
        assert self.nodes[0].n_moves == 0
        assert self.nodes[0].a_star_dist == 30

    def test__swap_copy(self):
        copy_node = copy.deepcopy(self.nodes[0])
        new_node = copy_node._swap_copy(2, 9)
        # Check the swap copy hasn't changed the original node
        assert copy_node == self.nodes[0]
        assert new_node.data == [2, 1, 11, 9, 5, 7, 3, 8, 6, 10, 4, 0]

    def test_slide(self):
        s = self.nodes[1]
        assert s.data == [5, 0, 2, 1, 4, 3]
        s.slide("U")
        assert s.data == [2, 0, 5, 1, 4, 3]
        assert s.n_moves == 1
        assert s.dist_from_solution(s.target_positions) == 2
        assert self.nodes[1].a_star_dist == 3

    def test_a_star_dist(self):
        s = self.nodes[2]
        assert self.nodes[2].a_star_dist == self.nodes[2].dist_from_solution()
        s.slide("R")
        d = self.nodes[2].dist_from_solution()
        assert self.nodes[2].a_star_dist == d + 1
        s.move_weight = 0.5
        s.slide("L")
        s.slide("R")
        assert self.nodes[2].a_star_dist == d + 1.5

    def test_return_neighbours(self):
        s = self.nodes[0]
        neighbours = s.return_neighbours(fixed_positions={0, 4, 8})
        assert len(neighbours) == 2
        assert neighbours[0].data == [2, 1, 10, 9, 5, 7, 3, 8, 6, 4, 11, 0]


class TestPathFinder:
    @pytest.fixture(autouse=True)
    def _create_node(self, slider_node):
        self.nodes = slider_node

    def test_show_route(self):
        s_0 = self.nodes[2]
        opt_path = 'LDRULURDLDRURULDRDLUURDLUL'
        s_end = slider.show_route(s_0, opt_path)
        assert s_end.dist_from_solution() == 0
        assert s_end.n_moves == len(opt_path)

    def test_find_a_star_path_full(self):
        """ Use a full A* search to find the solution to a 3 by 3 slider """
        s_end = find_a_star_path(self.nodes[2])
        assert s_end.path == 'LDRULURDLDRURULDRDLUURDLUL'
        assert s_end.dist_from_solution() == 0
        assert s_end.n_moves == 26

    def test_find_a_star_path_first_three(self):
        s_root = self.nodes[2]
        s_1 = find_a_star_path(s_root, target_values={0, 1, 2})
        assert s_1.dist_from_solution(target_values={0, 1, 2}) == 0
        assert s_1.dist_from_solution() == 8
        assert s_1.path == 'LURRDDLULDRUURDDLU'
        assert s_1.n_moves == 18

    def test_find_a_star_path_second_step(self):
        s_root = self.nodes[2]
        s_1 = find_a_star_path(s_root, target_values={0, 1, 2})
        s_2 = find_a_star_path(s_1, fixed_cells={0, 1, 2})
        # assert s_2.path == 'LURRDDLULDRUURDDLUULDRRULDLURRDLLU'
        assert s_2.path == 'LURRDDLULDRUURDDLULURDRULLDRURDLUL'
        assert s_2.dist_from_solution() == 0
        assert s_2.n_moves == 34

    def test_find_full_route_three_by_three_two_steps(self):
        s_root = self.nodes[2]

        s_2, timings = find_full_route(s_root)
        # assert s_2.path == 'LURRDDLULDRUURDDLUULDRRULDLURRDLLU'
        assert s_2.path == 'LURRDDLULDRUURDDLULURDRULLDRURDLUL'
        assert s_2.dist_from_solution() == 0
        assert s_2.n_moves == 34

    def test_find_full_route_three_by_three_three_steps(self):
        s_root = self.nodes[2]
        s_2, timings = find_full_route(s_root)
        assert s_2.dist_from_solution() == 0
        assert s_2.n_moves >= 34
        # assert s_2.path == "LURRDDLULDRUURDDLUULDRRULDLURRDLLU"
        assert s_2.path == "LURRDDLULDRUURDDLULURDRULLDRURDLUL"

    def test_find_full_route_four_by_four(self):
        s_root = self.nodes[3]
        s_end, timings = find_full_route(s_root)
        print(s_end, sum(timings))
        assert s_end.dist_from_solution() == 0

    def test_find_full_random(self):
        for _ in range(10):
            s_root = SliderNode(4)
            s_root.shuffle()
            s_end, timings = find_full_route(s_root)
            print(s_end, f'Time = {sum(timings):.3f} seconds')
            assert s_end.dist_from_solution() == 0

    def test_find_full_big(self):
        s_root = SliderNode(8)
        s_root.shuffle()
        s_end, timings = find_full_route(s_root)
        print(s_end, f'Time = {sum(timings):.3f} seconds')
        assert s_end.dist_from_solution() == 0


class TestHelperFunc:

    def test_slider_dist(self):
        assert slider.slider_dist(4, 13, 4) == 3
        assert slider.slider_dist(1, 18, 5) == 5

    def test_find_sub_region(self):
        assert slider.find_sub_region({6, 12, 13}, 4, 5) == {6, 7, 8, 11, 12, 13, 16, 17, 18}
        assert slider.find_sub_region({6, 12, 13}, 3, 5) == {1, 2, 3, 6, 7, 8, 11, 12, 13}
        assert slider.find_sub_region({7, 23}, 5, 5) == {7, 8, 9, 12, 13, 14, 17, 18, 19, 22, 23, 24}

    def test_create_partition(self):
        assert create_partition({9, 10, 11, 13, 14, 15}, 4, 4) == [{9, 10, 11, 13, 14, 15}]
        assert create_partition({2, 7, 12, 17}, 4, 5) == [{2}, {7, 12, 17}]
        assert create_partition({5, 6, 7, 9, 10, 11, 13, 14, 15}, 4, 4) == [{5, 6, 7}, {9, 10, 11, 13, 14, 15}]
        assert create_partition(set(range(25)), 5, 5) == [{0, 1}, {2, 3, 4}, {5}, {10, 15, 20}, {6}, {7, 8, 9},
                                                          {11, 16, 21}, {12, 13, 14}, {17, 18, 19, 22, 23, 24}]
