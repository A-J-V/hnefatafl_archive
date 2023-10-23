import numpy as np


# This is a global constant that maps piece planes to team
TEAMS = {0: 1, 1: 2, 2: 2}


# Several small convenience functions that are used in multiple places for condition checks
def is_piece(board_array, row, column):
    # This could cause an index out of bounds error!
    return board_array[:, row, column].any()


def is_edge(row, col, size):
    return row == 0 or col == 0 or row == size or col == size


def is_blank(board_array, row, col):
    size = board_array.shape[-1] - 1
    return (0 <= row <= size and 0 <= col <= size) and board_array[:, row, col].any()


def near_blank(board_array, row, col):
    return (is_blank(board_array, row - 1, col) or is_blank(board_array, row + 1, col) or
            is_blank(board_array, row, col - 1) or is_blank(board_array, row, col + 1))


def is_ally(board_array, row, column, ally):
    return is_piece(board_array, row, column) and TEAMS[np.argwhere(board_array[:, row, column] == 1).item()] == ally
# The group of small convenience functions ends here


def get_moves(board_array: np.array,
              index: tuple,
              ) -> np.array:
    """
    Return a binary array of legal moves.

    The index of the array encodes a move relative to the passed in index.
    The value of the array indicates whether that move is legal (1) or not (0).
    Indices 0-9 encode "move up 1-10 spaces", indices 10-19 encode "move down 1-10 spaces",
    Indices 20-29 encode "move left 1-10 spaces", indices 30-39 encode "move right 1-10 spaces".

    :param np.array board_array: a 2D NumPy array representing the board
    :param tuple index: a tuple(int, int) representing the index of the piece whose legal moves we're checking.
    :return: A 1D binary NumPy array of length 40 representing legal moves
    """
    size = board_array.shape[-1]
    legal_moves = np.zeros(40)
    dirty_indices = []
    # If there is no piece on this tile, return immediately (there are no legal moves if there's no piece!)
    if not board_array[:, index[0], index[1]].any():
        return legal_moves
    # Need to go through the 40 possible moves and check the legality of each...
    # 0-9 is up 1-10, 10-19 is down 1-10, 20-29 is left 1-10, 30-39 is right 1-10
    # Two safety checks are necessary to prevent out of bounds errors.
    safe = {-1: lambda x: x > 0, 1: lambda x: x < (size - 1)}

    # Directions are encoded as (row/column, increment/decrement) where row=0, column=1, increment=1, decrement=-1
    for k, instruction in enumerate([(0, -1), (0, 1), (1, -1), (1, 1)]):
        axis, direction = instruction
        tmp_index = list(index)
        i = k * 10
        while i < (k + 1) * 10 and safe[direction](tmp_index[axis]):
            tmp_index[axis] += direction
            if not board_array[:, tmp_index[0], tmp_index[1]].any():
                # No blocking piece
                legal_moves[i] = 1
            elif board_array[:, tmp_index[0], tmp_index[1]].any():
                # Blocking piece
                dirty_indices.append((tmp_index[0], tmp_index[1]))
                break
            i += 1
    return legal_moves


def make_move(board_array: np.array,
              index: tuple,
              move: int,
              ) -> None:
    """Move the piece at index according to move. Assumes the move is legal."""
    # Find which plane the piece is on (which piece type it is)
    plane = np.argwhere(board_array[:, index[0], index[1]] == 1).item()
    # Get the move axis, direction, and number of tiles
    axis = 0 if move < 20 else 1
    direction = 1 if move >= 30 or (20 > move >= 10) else -1
    num = (move % 10) + 1
    # Move the move to the new index and set the old index to 0
    new_index = list(index)
    new_index[axis] += direction * num
    board_array[plane, new_index[0], new_index[1]] = 1
    board_array[plane, index[0], index[1]] = 0


def get_nice_variables(board_array: np.array,
                       index: tuple,
                       ) -> tuple:
    """Return some convenient variables used in multiple game utilities"""
    row, col = index
    teams = {0: 1, 1: 2, 2: 2}
    size = board_array.shape[-1] - 1
    hostile = [(0, 0), (0, size), (size, 0), (size, size)]
    plane = np.argwhere(board_array[:, index[0], index[1]] == 1).item()
    ally = teams[plane]
    return row, col, teams, size, hostile, plane, ally


def check_capture(board: np.array,
                  index: tuple,
                  ) -> None:
    """Given an index, checks to see if any basic enemies pieces around it are captured."""
    # Set up some convenient variables
    row, col, teams, size, hostile, plane, ally = get_nice_variables(board, index)

    # If the throne is empty, it is hostile
    if not board[:, size // 2, size // 2].any():
        hostile.append((size // 2, size // 2))

    # Set up some convenient anonymous functions to check conditions
    is_enemy = lambda r, c: (is_piece(board, r, c) and
                             np.argwhere(board[:, r, c] == 1).item() not in [plane, 2])
    is_flanked = lambda r, c: ((is_piece(board, r, c) and
                                teams[np.argwhere(board[:, r, c] == 1).item()] == ally) or
                               (r, c) in hostile)
    is_edge = lambda r, c: r == 0 or c == 0 or r == size or c == size

    # All of these if statements could probably be collapsed in a similar way as check_shield_wall()
    if row > 0 and is_enemy(row - 1, col):
        if is_edge(row - 1, col):
            tags = []
            if check_shield_wall(board, (row - 1, col), tags):
                capture_tags(board, tags)
        # if the enemy is not on an edge, and the other side is an allied piece or hostile piece
        if row - 2 >= 0 and is_flanked(row - 2, col):
            # Destroy it!
            board[:, row - 1, col] = 0

    if row < size and is_enemy(row + 1, col):
        if is_edge(row + 1, col):
            tags = []
            if check_shield_wall(board, (row + 1, col), tags):
                capture_tags(board, tags)
        if row + 2 <= size and is_flanked(row + 2, col):
            board[:, row + 1, col] = 0

    if col > 0 and is_enemy(row, col - 1):
        if is_edge(row, col - 1):
            tags = []
            if check_shield_wall(board, (row, col - 1), tags):
                capture_tags(board, tags)
        if col - 2 >= 0 and is_flanked(row, col - 2):
            board[:, row, col - 1] = 0

    if col < size and is_enemy(row, col + 1):
        if is_edge(row, col + 1):
            tags = []
            if check_shield_wall(board, (row, col + 1), tags):
                capture_tags(board, tags)
        if col + 2 <= size and is_flanked(row, col + 2):
            board[:, row, col + 1] = 0


def capture_tags(board_array: np.array,
                 tags: list,
                 ) -> None:
    for tag in tags:
        if np.argwhere(board_array[:, tag[0], tag[1]] == 1).item() != 2:
            board_array[:, tag[0], tag[1]] = 0


def check_shield_wall(board: np.array,
                      index: tuple,
                      tags: list,
                      edge: str='',
                      ) -> bool:
    """Recursively check whether a shield wall capture can be executed."""
    row, col, teams, size, hostile, plane, ally = get_nice_variables(board, index)

    if not edge:
        if row == 0:
            edge = 'up'
        elif row == size:
            edge = 'down'
        elif col == 0:
            edge = 'left'
        else:
            edge = 'right'

    is_ally = lambda r, c: is_piece(board, r, c) and teams[np.argwhere(board[:, r, c] == 1).item()] == ally
    not_blank = lambda r, c: board[:, r, c].any() or (r, c) in hostile
    is_hostile = lambda r, c: ((is_piece(board, r, c) and
                                teams[np.argwhere(board[:, r, c] == 1).item()] != ally) or
                               (r, c) in hostile)

    h_mapping = {'up': (row + 1, col), 'down': (row - 1, col), 'left': (row, col + 1), 'right': (row, col - 1)}
    b_mapping = {'up': ((row, col - 1), (row, col + 1)), 'down': ((row, col - 1), (row, col + 1)),
                 'left': ((row - 1, col), (row + 1, col)), 'right': ((row - 1, col), (row + 1, col))}
    h_dir = h_mapping[edge]
    b_dirs = b_mapping[edge]

    if (is_hostile(h_dir[0], h_dir[1]) and
        not_blank(b_dirs[0][0], b_dirs[0][1]) and
        not_blank(b_dirs[1][0], b_dirs[1][1])
    ):
        tags.append(index)
        adjacent_friends = []
        if is_ally(b_dirs[0][0], b_dirs[0][1]) and (b_dirs[0][0], b_dirs[0][1]) not in tags:
            adjacent_friends.append((b_dirs[0][0], b_dirs[0][1]))
        if is_ally(b_dirs[1][0], b_dirs[1][1]) and (b_dirs[1][0], b_dirs[1][1]) not in tags:
            adjacent_friends.append((b_dirs[1][0], b_dirs[1][1]))
        for ally in adjacent_friends:
            if not check_shield_wall(board, ally, tags, edge):
                return False
    else:
        return False
    return True


def check_exit_fort(board_array: np.array) -> bool:
    """Recursively check whether defenders have built an Exit Fort."""
    row, col = tuple(np.argwhere(board_array[2, :, :] == 1)[0])
    size = board_array.shape[-1]
    if not (is_edge(row, col, size) and near_blank(board_array, row, col)):
        return False

    return True


def check_king(board_array: np.array,
               ) -> int:
    """
    Check whether the King has escaped or been captured.

    :param np.array board_array: A 2D NumPy array representing the board
    :return: -1 means King captured, 1 means King escaped, 0 means neither.
    """
    size = board_array.shape[-1] - 1
    row, col = tuple(np.argwhere(board_array[2, :, :] == 1)[0])
    corners = [(0, 0), (0, size), (size, 0), (size, size)]
    throne = (size / 2, size / 2)
    teams = {0: 1, 1: 2, 2: 2}
    is_hostile = lambda row, col: (row, col) == throne or (
            board_array[:, row, col].any() and teams[np.argwhere(board_array[:, row, col] == 1).item()] != 2)

    # Has the King escaped?
    if (row, col) in corners:
        return 1

    # Is the king surrounded?
    if ((row - 1 > 0     and is_hostile(row - 1, col)) and
        (row + 1 <= size and is_hostile(row + 1, col)) and
        (col - 1 > 0     and is_hostile(row, col - 1)) and
        (col + 1 <= size and is_hostile(row, col + 1))
       ):
        return -1
    return 0
