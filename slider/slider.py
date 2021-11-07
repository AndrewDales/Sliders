import copy
import heapq
from random import shuffle
from functools import lru_cache
from numpy import array
import numpy as np
from time import perf_counter


class SlideError(Exception):
    pass


class Slider:
    def __init__(self, n_rows=None, n_cols=None, order=None, solution=None):
        """Slider can be initialised with an order - a square number permutation of values or if n_dims is specified
        the order will be the range 0 to n_dims**2 - 1"""
        if n_rows is None:
            n_rows = 4
        if n_cols is None:
            n_cols = n_rows
        if order is None:
            order = list(range(n_cols * n_rows))
        if solution is None:
            solution = tuple(sorted(order))

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.data = list(order)
        self.solution = solution
        self.blank = max(self.data)
        self.blank_pos = self.data.index(self.blank)

    @property
    def coords(self):
        return tuple(divmod(val, self.n_cols) for val in self.data)

    @property
    def is_solved(self):
        """ Check if the slider is in a solved position"""
        return tuple(self.data) == self.solution

    @property
    def is_soluble(self):
        """Find if the permutation is soluble by finding the parity of the permutation + the parity of the
        taxicab distance of the blank square in the problem and the taxi distance of the blank square in the solution.
        If the two are equal - the solution is soluble
        see https://en.wikipedia.org/wiki/15_puzzle """
        parity = self._find_parity(self.data, self.solution)
        blank_val = max(self.data)
        blank_data_parity = sum(self._find_row_col(self.data.index(blank_val), self.n_cols)) % 2
        blank_solution_parity = sum(self._find_row_col(self.solution.index(blank_val), self.n_cols)) % 2

        return (parity + blank_data_parity + blank_solution_parity) % 2 == 0

    @staticmethod
    def _find_row_col(val, n_cols):
        """ Find the row and column given an index value. Default is the blank position """
        return array(divmod(val, n_cols))

    @staticmethod
    def _find_pos_index(coords, n_cols):
        """ Find the position of an element given its row and column"""
        return coords[0] * n_cols + coords[1]

    @staticmethod
    def _find_parity(perm_1, perm_2):
        """ Whether perm_1 and perm_2 have the same parity i.e. if there are an even number of swaps from perm_1 to
        perm_2, the parity is 0, if there are a odd number of swaps the parity is 1"""
        parity = 0
        trial = copy.copy(perm_1)
        for i in range(len(trial) - 1):
            # if element i is not in the correct place, swap it with the correct element
            p, q = trial[i], perm_2[i]
            if p != q:
                j = trial.index(q)
                trial[i], trial[j] = trial[j], trial[i]
                parity += 1
        return parity % 2

    def _swap(self, i1, i2):
        """ Swap the positions of two values in the Slider data"""
        self.data[i1], self.data[i2] = self.data[i2], self.data[i1]
        if i2 == self.blank_pos:
            self.blank_pos = i1

    def find_blank_neighbours(self, fixed_cells=None):
        """Find all the possible neighbours for the blank position only allowing neighbours that are in the
        movable_rows or movable_cols lists.
        Default is that all rows and columns are movable"""
        if fixed_cells is None:
            fixed_cells = set()

        blank_pos = self._find_row_col(self.blank_pos, self.n_cols)
        dirs = {"R": array((0, -1)), "L": array((0, 1)), "U": array((1, 0)), "D": array((-1, 0))}
        neighbours = {}
        for k, direction_vector in dirs.items():
            cell_pos = blank_pos + direction_vector
            # Only allow neighbours which are within the grid
            if ((0 <= cell_pos[0] < self.n_rows) and (0 <= cell_pos[1] < self.n_cols)
                    and not (self._find_pos_index(cell_pos, self.n_cols) in fixed_cells)):
                neighbours[k] = self._find_pos_index(cell_pos, self.n_cols)
        return neighbours

    def shuffle(self):
        """ Permute the order of the permutations and checks they are valid solutions """
        shuffle(self.data)

        # If new position is not soluble then swap the first two element
        if not self.is_soluble:
            i_1 = self.data.index(self.solution[0])
            i_2 = self.data.index(self.solution[1])
            self._swap(i_1, i_2)
        self.blank_pos = self.data.index(max(self.data))

    def slide(self, direction):
        """ Slide 'R','L', 'U' or 'D' to move a slider tile into the blank space"""
        # Find the neighbours of the blank cell, which are tiles that can be moved into the blank space
        neighbours = self.find_blank_neighbours()
        # Find the tile that can be moved 'R', 'L', 'U' or 'D'
        try:
            cell = neighbours[direction]
            # cell_index = self._find_pos_index(cell, self.n_cols)
        # Raise an error if there is no cell that can be moved in the desired swap_direction
        except KeyError:
            raise SlideError(f"Can not slide in direction '{direction}'")
        # Swap the relevant cell with the blank position
        self._swap(cell, self.blank_pos)

    def dist_from_solution(self, target_values=None):
        """ Find the total distance from the current positions to the solution position, looking only
        at the cells in target_values. The distance is the taxicab distance of each cell from its desired
        location """
        if target_values is None:
            target_values = set(self.data)

        data_i = [self.data.index(i) for i in target_values]
        sol_i = [self.solution.index(i) for i in target_values]

        return sum(slider_dist(x, y, self.n_cols) for x, y in zip(data_i, sol_i))

    def __repr__(self):
        data = self.data.copy()
        data[self.blank_pos] = "_"
        # Create a print template, as a list of n_dims * n_dims '{}' suitably spaced.
        form_temp = "\n".join([" ".join(['{:<3}'] * self.n_cols)] * self.n_rows)
        return form_temp.format(*data)

    def __eq__(self, other):
        return self.data == other.data


class SliderNode(Slider):
    """ SliderNode is a subclass of Slider used in route finding. It includes a set of target positions,
    which are the positions that are trying to be solved in this round and an a-star move_weight.
    A higher a-star move_weight will mean that the a_star measure puts a higher weight on
    the distance that the slider needs to move and a lower weight on the moves already taken.
    A higher a-star move_weight will mean that an optimal solution will tend to take fewer moves,
    but will take longer to find."""
    def __init__(self, n_rows=None, n_cols=None, order=None, solution=None,
                 target_positions=None,
                 move_weight=0.5):
        super().__init__(n_rows, n_cols, order, solution)
        if target_positions is None:
            target_positions = self.data

        self.target_positions = target_positions
        self.n_moves = 0
        self.path = ""
        self.move_weight = move_weight

    @property
    def a_star_dist(self):
        """ Returns the 'a_star distance' this is a combination of the distance from solution heuristic
        and the number of moves taken so far. """
        return (self.dist_from_solution(self.target_positions) +
                self.n_moves * self.move_weight)

    def __gt__(self, other):
        """ Use the a_stat_dist to order SliderNodes - higher a_stat_dist are larger nodes.
        This is used to sort the SliderNodes in a priority queue"""
        return self.a_star_dist > other.a_star_dist

    def _swap_copy(self, i1, i2):
        """ Return slider object with two items swapped"""
        data = copy.copy(self.data)
        data[i1], data[i2] = data[i2], data[i1]
        new_slider = SliderNode(self.n_rows, self.n_cols, data,
                                self.solution, self.target_positions,
                                self.move_weight)
        return new_slider

    def slide(self, direction):
        """ Modified slide keeps track of movement path"""
        super().slide(direction)
        self.n_moves += 1
        self.path += direction

    def shuffle(self):
        """ modified shuffle, resets path and n_moves"""
        super().shuffle()
        self.n_moves = 0
        self.path = ""

    def return_neighbours(self, fixed_positions=None):
        """ return a new set of SliderNodes which are neighbours of self and are in the allowable rows or columns"""
        if fixed_positions is None:
            fixed_positions = set()

        neighbour_cells = self.find_blank_neighbours(fixed_cells=fixed_positions)
        neighbour_sliders = []
        for direction, pos_index in neighbour_cells.items():
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


def find_a_star_path(starting_node: SliderNode,
                     target_values=None,
                     fixed_cells=None):
    """ Use A* algorithm to find a path from the starting_node's current position to its solution.
        Only take into account the positions of the values in target_values
        The heuristic function is the sum of the Manhattan distance for each member of target_values from its
        position in solution """
    if target_values is None:
        target_values = set(range(starting_node.n_rows * starting_node.n_cols))
    if fixed_cells is None:
        fixed_cells = set()

    current_node = copy.deepcopy(starting_node)
    current_fixed = copy.copy(fixed_cells)
    current_node.target_positions = target_values
    visited = set()
    priority_queue = []
    sub_size = max(min(current_node.n_cols, current_node.n_rows) - 1, 3)
    distant_region = set()

    # Find the three-by-three (or larger) zone containing the targets
    # distant_region is the rest of the board, which can be fixed when the target cells are not in it
    close_region = find_sub_region(target_values, current_node.n_rows, current_node.n_cols, sub_size)
    distant_region = set(current_node.solution) - close_region

    heapq.heappush(priority_queue, current_node)
    # todo - keeping track of distance. Remove when tested
    dist = current_node.dist_from_solution(target_values)

    while current_node.dist_from_solution(target_values) > 0 and priority_queue:
        # Take the smallest node from the priority_queue (the size of the node depends on its A* distance from the
        # solution taking into account only the values in target_values)
        current_node = heapq.heappop(priority_queue)

        # Check if all the target cells in the current mode are outside the distant_region
        if distant_region:
            cells_in_distant = {current_node.data[i] for i in distant_region}
            if not (cells_in_distant & (target_values | {current_node.blank})):
                current_fixed = distant_region | fixed_cells
                # print("***Fixing***")
                # print(f"New sub size = {sub_size}")
                # print(current_node)
                sub_size -= 1
                if sub_size >= 3:
                    close_region = find_sub_region(target_values, current_node.n_rows, current_node.n_cols, sub_size)
                    distant_region = set(current_node.solution) - close_region
                else:
                    distant_region = set()
                priority_queue = []

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
        neighbours = current_node.return_neighbours(fixed_positions=current_fixed)
        for ngh in neighbours:
            if tuple(ngh.data) in visited:
                continue
            heapq.heappush(priority_queue, ngh)

    if current_node.dist_from_solution(target_values) != 0:
        raise RuntimeError("Could not find solution path")

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
    partition = create_partition(set(range(slider.n_rows * slider.n_cols)), slider.n_rows, slider.n_cols)
    s = [copy.deepcopy(slider)]
    fixed = set()
    times = []
    t = perf_counter()
    for current_part in partition:
        s_step = find_a_star_path(s[-1], target_values=current_part, fixed_cells=fixed)
        times.append(perf_counter() - t)
        t = perf_counter()
        # print(s_step)
        s.append(copy.deepcopy(s_step))
        fixed |= current_part
    return s[-1], times


@lru_cache
def slider_dist(x, y, num_cols):
    x1, x2 = divmod(x, num_cols)
    y1, y2 = divmod(y, num_cols)
    return abs(x1 - y1) + abs(x2 - y2)


def find_sub_region(targets, n_rows, n_cols, sub_size=3):
    """ Find the sub-region containing all the cells in targets, which is at least 3 by 3"""
    rows, cols = np.divmod(tuple(targets), n_cols)
    r_min = min(min(rows), n_rows - sub_size)
    r_max = max(r_min + sub_size, max(rows) + 1)
    c_min = min(min(cols), n_cols - sub_size)
    c_max = max(c_min + sub_size, max(cols) + 1)
    return {i * n_cols + j for i in range(r_min, r_max) for j in range(c_min, c_max)}


def create_partition(initial_set, n_rows, n_cols):
    """ Recursive function to create a partition by splitting off the first row or column
    Single rows or columns are split into chunks of three or less """

    rows, cols = np.divmod(tuple(initial_set), n_cols)
    row_range = max(rows) - min(rows) + 1
    col_range = max(cols) - min(cols) + 1

    # smaller than 3*2 chunks arrangements don't have to be split
    if (row_range <=2 and col_range <=3) or (row_range <= 3 and col_range <=2):
        partitioned_set = [initial_set]

    # Split the list into chunks of three from the end of the list
    elif row_range == 1 or col_range == 1:
        sorted_set = sorted(initial_set, reverse=True)
        partitioned_set = [set(sorted_set[i:i+3]) for i in range(0, len(sorted_set), 3)]
        partitioned_set.reverse()

    # Split off the top row
    elif row_range >= 3 and row_range >= col_range:
        first_row = {i for i, r in zip(initial_set, rows) if r == min(rows)}
        remainder = initial_set - first_row
        partitioned_set = create_partition(first_row, n_rows, n_cols) + create_partition(remainder, n_rows, n_cols)

    # Split off the left-most column
    else:
        first_col = {i for i, c in zip(initial_set, cols) if c == min(cols)}
        remainder = initial_set - first_col
        partitioned_set = create_partition(first_col, n_rows, n_cols) + create_partition(remainder, n_rows, n_cols)

    return partitioned_set


if __name__ == '__main__':
    # s_0 = SliderNode(4, 4, [14, 2, 6, 10, 15, 12, 4, 8, 11, 5, 13, 7, 3, 9, 1, 0])
    # tic = perf_counter()
    # s_end, timings = find_full_route(s_0, [{0, 1, 2, 3}, {4, 8, 12}, {5, 6, 7}, {9, 10, 11, 13, 14, 15}])
    # s_end, timings = find_full_route(s_0, [{0, 1, 4, 5}, {2, 3, 6, 7}, {8, 9, 12, 13}, {10, 11, 14, 15}])
    # s_end, timings = find_full_route(s_0, [{0, 1}, {2, 3}, {4, 8, 12}, {5, 6, 7}, {9, 10, 11, 13, 14, 15}])
    # print(s_end[-1])
    # print(f'Time taken = {sum(timings):.2f} seconds')
    # find_a_star_path(SliderNode(4, 4, [0, 15, 15, 7, 3, 2, 12, 11, 9, 13, 5, 1, 6, 8, 10, 4]), target_values={1, 2, 3},
    #                  fixed_cells={0}, allow_fixing=True)

    s_0 = SliderNode(7)
    s_0.shuffle()
    s_end, timings = find_full_route(s_0)
    print(f'Time taken = {sum(timings):.2f} seconds')
