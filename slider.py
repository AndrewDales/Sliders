import copy
import heapq
from random import randint

import numpy as np
from numpy import array


class SlideError(Exception):
    pass


class Slider:
    def __init__(self, n_rows=None, n_cols=None, order=None, solution=None):
        """Slider can be initialised with an order - a square number permutation of values or if n_dims is specified
        the order will be the range 0 to n_dims**2 - 1"""
        if order is None:
            order = list(range(n_cols * n_rows))
        if solution is None:
            solution = tuple(sorted(order))

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.data = list(order)
        self.solution = solution
        self.blank_pos = self.data.index(max(self.data))

    @staticmethod
    def _find_row_col(val, n_cols):
        """ Find the row and column given an index value. Default is the blank position """
        return array(divmod(val, n_cols))

    @staticmethod
    def _find_pos_index(coords, n_cols):
        """ Find the position of an element given its row and column"""
        return coords[0] * n_cols + coords[1]

    def _swap(self, i1, i2):
        """ Swap the positions of two values in the Slider data"""
        self.data[i1], self.data[i2] = self.data[i2], self.data[i1]
        if i2 == self.blank_pos:
            self.blank_pos = i1

    def find_blank_neighbours(self, movable_rows=None, movable_cols=None):
        """Find all the possible neighbours for the blank position only allowing neighbours that are in the
        movable_rows or movable_cols lists.
        Default is that all rows and columns are movable"""
        if movable_rows is None:
            movable_rows = tuple(range(self.n_rows))
        if movable_cols is None:
            movable_cols = tuple(range(self.n_cols))

        blank_pos = self._find_row_col(self.blank_pos, self.n_cols)
        dirs = {"R": array((0, -1)), "L": array((0, 1)), "U": array((1, 0)), "D": array((-1, 0))}
        neighbours = {}
        for k, direction_vector in dirs.items():
            cell_pos = blank_pos + direction_vector
            # Only allow neighbours which are within the grid
            if cell_pos[0] in movable_rows and cell_pos[1] in movable_cols:
                neighbours[k] = cell_pos
        return neighbours

    def shuffle(self):
        """ Permute the order ensuring that only even parity permutations are allowed"""
        parity = 0

        # Use the Fisher and Yates' shuffle - keep track of the parity.
        # Parity can be either odd or even and each swap changes the parity by 1
        for i in range(len(self.data) - 1):
            j = randint(i, len(self.data) - 1)
            self._swap(i, j)
            parity = (parity + i + j) % 2

        self.blank_pos = self.data.index(max(self.data))
        # The parity of the blank + the parity of all the swaps must be even. If necessary, make another swap
        parity_blank = sum(self._find_row_col(self.blank_pos, self.n_cols)) % 2
        if parity + parity_blank % 2 == 1:
            self._swap(0, 1)

    def slide(self, direction):
        """ Slide 'R','L', 'U' or 'D' to move a slider tile into the blank space"""
        # Find the neighbours of the blank cell, which are tiles that can be moved into the blank space
        neighbours = self.find_blank_neighbours()
        # Find the tile that can be moved 'R', 'L', 'U' or 'D'
        try:
            cell = neighbours[direction]
            cell_index = self._find_pos_index(cell, self.n_cols)
        # Raise an error if there is no cell that can be moved in the desired swap_direction
        except KeyError:
            raise SlideError(f"Can not slide in direction '{direction}'")
        # Swap the relevant cell with the blank position
        self._swap(cell_index, self.blank_pos)

    def dist_from_solution(self, target_values=None):
        """ Find the total distance from the current positions to the solution position"""
        if target_values is None:
            target_values = self.data

        data_i = [self.data.index(i) for i in target_values]
        sol_i = [self.solution.index(i) for i in target_values]

        current_coords = np.divmod(data_i, self.n_cols)
        sol_coords = np.divmod(sol_i, self.n_cols)

        # The distance is the sum of the absolute distances between the coords of the current position
        # and the coordinates of the solution position
        return sum(abs(current_coords[0] - sol_coords[0]) + abs(current_coords[1] - sol_coords[1]))


    def __repr__(self):
        data = self.data.copy()
        data[self.blank_pos] = "_"
        # Create a print template, as a list of n_dims * n_dims '{}' suitably spaced.
        form_temp = "\n".join([" ".join(['{:<3}'] * self.n_cols)] * self.n_rows)
        return form_temp.format(*data)

    def __eq__(self, other):
        return self.data == other.data


class SliderNode(Slider):
    def __init__(self, n_rows=None, n_cols=None, order=None, solution=None, target_positions=None):
        super().__init__(n_rows, n_cols, order, solution)
        if target_positions is None:
            target_positions = self.data

        self.target_positions = target_positions
        self.n_moves = 0
        self.path = ""

    @property
    def a_star_dist(self):
        return self.dist_from_solution(self.target_positions) + self.n_moves

    def __gt__(self, other):
        return self.a_star_dist > other.a_star_dist

    def _swap_copy(self, i1, i2):
        """ Return slider object with two items swapped"""
        data = copy.copy(self.data)
        data[i1], data[i2] = data[i2], data[i1]
        new_slider = SliderNode(self.n_rows, self.n_cols, data, self.solution, self.target_positions)
        return new_slider

    def slide(self, direction):
        """ Modified slide keeps track of movement path"""
        super().slide(direction)
        self.n_moves += 1
        self.path += direction

    def return_neighbours(self, allowable_rows, allowable_columns):
        """ return a new set of SliderNodes which are neighbours of self and are in the allowable rows or columns"""
        neighbour_cells = self.find_blank_neighbours(allowable_rows, allowable_columns)
        neighbour_sliders = []
        for direction, cell in neighbour_cells.items():
            # Find the position index of the neighbour
            pos_index = self._find_pos_index(cell, self.n_cols)

            # User _swap_copy to create copies of each of the neighbour cells.
            # Change the n_moves and path attributes to show the extra move
            neighbour = self._swap_copy(pos_index, self.blank_pos)
            neighbour.n_moves = self.n_moves + 1
            neighbour.path = self.path + direction
            neighbour_sliders.append(neighbour)
        return neighbour_sliders

    def __repr__(self):
        repr_str = super().__repr__()
        repr_str += f'\nmoves = {self.n_moves} distance = {self.dist_from_solution(self.target_positions)}\n'
        return repr_str


def find_a_star_path(starting_node: SliderNode, target_values=None, rows=None, cols=None):
    """ Use A* algorithm to find a path from the starting_node's current position to its solution.
        Only take into account the positions of the values in target_values
        The heuristic function is the sum of the Manhattan distance for each member of target_values from its
        position in solution """
    if rows is None:
        rows = tuple(range(starting_node.n_rows))
    if cols is None:
        cols = tuple(range(starting_node.n_cols))
    if target_values is None:
        target_values = tuple(range(starting_node.n_rows*starting_node.n_cols))

    current_node = copy.deepcopy(starting_node)
    current_node.target_positions = target_values
    visited = set()
    priority_queue = []

    heapq.heappush(priority_queue, starting_node)
    # todo - keeping track of distance. Remove when tested
    dist = current_node.dist_from_solution(target_values)

    while current_node.dist_from_solution(target_values) > 0:
        # Take the smallest node from the priority_queue (the size of the node depends on its A* distance from the
        # solution taking into account only the values in target_values)
        current_node = heapq.heappop(priority_queue)

        # ToDo - check that distance is decreasing, remove when tested
        new_dist = current_node.dist_from_solution(target_values)
        if new_dist < dist:
            dist = new_dist
            # print(current_node)

        # move on to the next Node in the priority queue if the current_node position has already been visited
        if tuple(current_node.data) in visited:
            continue

        # Add the position of the current node to visited
        visited.add(tuple(current_node.data))

        # Find the neighbours of the current_node and push them onto the priority queue if they have been
        neighbours = current_node.return_neighbours(rows, cols)
        for ngh in neighbours:
            if tuple(ngh.data) in visited:
                continue
            heapq.heappush(priority_queue, ngh)

    return current_node


def show_route(starting_node: SliderNode, path):
    """ Print out the route given a starting point and a path"""
    current_node = copy.deepcopy(starting_node)

    print(current_node)
    for letter in path:
        current_node.slide(letter)
        print(current_node)
    return current_node


def find_full_route(slider):
    pass


if __name__ == '__main__':
    # initial_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 12, 15]
    # # initial_order = [4, 5, 1, 13, 3, 9, 12, 2, 11, 15, 0, 14, 7, 8, 6, 10]
    # initial_position = [3, 0, 5, 4, 2, 1]
    # # initial_order = [0, 1, 2, 8, 3, 5, 6, 4, 7]
    # s_root = SliderNode(order=initial_order)
    # sol_path = find_a_star_path(initial_order)
    #
    # print("Showing Path".center(25, '*'))
    # print(s_root)
    #
    # for path_dir in sol_path:
    #     s_root.slide(path_dir)
    #     s_root.n_moves += 1
    #     print(s_root)
    # initial_position = [2, 8, 1, 0, 7, 4, 6, 3, 5]
    initial_position = [6, 5, 4, 3, 8, 2, 1, 7, 0]
    part = (0, 1, 2)
    s_root = SliderNode(3, 3, initial_position, target_positions=part)
    s_0 = copy.deepcopy(s_root)
    s_0.target_positions = tuple(range(8))
    s_1 = find_a_star_path(s_0, part)
    part = (3, 4, 5, 6, 7, 8)
    s_2 = find_a_star_path(s_1, part, rows=(1,2))
    show_route(s_root, s_2.path)
    s_straight = find_a_star_path(s_0)
