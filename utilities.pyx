from collections import deque
import cython
import numpy as np
cimport numpy as np
from typing import List, Tuple

# This is a global constant that maps piece planes to team
TEAMS = {0: 1, 1: 2, 2: 2}


# Several small convenience functions that are used in multiple places for condition checks
cdef is_piece(board: np.array, row: np.int64, col: np.int64):
    # This could cause an index out of bounds error!
    return ((board[0, row, col] == 1) or
            (board[1, row, col] == 1) or
            (board[2, row, col] == 1))


def is_king(board: np.array, row: np.int64, col: np.int64):
    return board[2, row, col] == 1


def find_king(board: np.array):
    return tuple(np.argwhere(board[2, :, :] == 1)[0])


def is_defender(board: np.array, row: np.int64, col: np.int64):
    return board[1, row, col] == 1


def is_attacker(board: np.array, row: np.int64, col: np.int64):
    return board[0, row, col] == 1


def is_edge(row, col, size):
    return row == 0 or col == 0 or row == size or col == size


def is_corner(row, col, size):
    return (row, col) == (0, 0) or (row, col) == (0, size) or (row, col) == (size, 0) or (row, col) == (size, size)


def in_bounds(row, col, size):
    return 0 <= row <= size and 0 <= col <= size


def is_blank(board, row, col):
    size = board.shape[-1] - 1
    return in_bounds(row, col, size) and not is_piece(board, row, col)


def near_blank(board_array, row, col):
    return (is_blank(board_array, row - 1, col) or is_blank(board_array, row + 1, col) or
            is_blank(board_array, row, col - 1) or is_blank(board_array, row, col + 1))


def is_ally(board_array, row, column, ally):
    return is_piece(board_array, row, column) and TEAMS[np.argwhere(board_array[:, row, column] == 1).item()] == ally


def is_hostile(board, row, col, ally, hostile):
    return ((is_piece(board, row, col) and TEAMS[np.argwhere(board[:, row, col] == 1).item()] != ally) or
            (row, col) in hostile)


# The group of small convenience functions ends here



cpdef get_moves(board: np.array,
                index: Tuple[int, int],
                cache: np.array,
                dirty_map: dict,
                dirty_flags: np.array,
                ) except *:
    """
    Return a binary array of legal moves.

    The index of the array encodes a move relative to the passed in index.
    The value of the array indicates whether that move is legal (1) or not (0).
    Indices 0-9 encode "move up 1-10 spaces", indices 10-19 encode "move down 1-10 spaces",
    Indices 20-29 encode "move left 1-10 spaces", indices 30-39 encode "move right 1-10 spaces".

    :param np.array board: a 2D NumPy array representing the board
    :param tuple index: a tuple(int, int) representing the index of the piece whose legal moves we're checking.
    :param np.array cache: a reference to an array of the cached action space
    :param dict dirty_map: a dict mapping index i to a list of indices whose caches are invalidate if i moves
    :param np.array dirty_flags: a 2D array of flags indicating whether the move cache of (row, col) is dirty
    :return: A 1D binary NumPy array of length 40 representing legal moves
    """
    # This involves unnecessary casting between lists and tuples and packed and unpacked.
    # It should be cleaned up to try to standardize how the index is to be used through the function (and related funcs)
    # If there is no piece on this tile, return immediately (there are no legal moves if there's no piece!)
    if not is_piece(board, index[0], index[1]):
        return np.zeros(40)
    # If the cache of this index is not dirty, immediately return the cached legal moves.
    elif index not in dirty_flags:
        return cache[:, index[0], index[1]]
    cdef int size
    size = board.shape[2] - 1
    restricted = [[0, 0], [0, size], [size, 0], [size, size], [size // 2, size // 2]]
    dirty_indices = []
    legal_moves = np.zeros(40)
    # Need to go through the 40 possible moves and check the legality of each...
    # 0-9 is up 1-10, 10-19 is down 1-10, 20-29 is left 1-10, 30-39 is right 1-10

    cdef int k
    cdef tuple instruction
    cdef int i
    cdef int axis
    cdef int direction
    cdef int dx[4]
    dx[0], dx[1], dx[2], dx[3] = 0, 0, 1, 1
    cdef int dy[4]
    dy[0], dy[1], dy[2], dy[3] = -1, 1, -1, 1
    cdef int initial_row
    cdef int initial_col
    initial_row = index[0]
    initial_col = index[1]
    cdef int tmp_index[2]

    for k in range(4):
        axis = dx[k]
        direction = dy[k]
        tmp_index[0] = initial_row
        tmp_index[1] = initial_col
        i = k * 10
        while i < (k + 1) * 10:
            tmp_index[axis] = tmp_index[axis] + direction
            if (tmp_index[0] < 0) or (tmp_index[0] > size) or (tmp_index[1] < 0) or (tmp_index[1] > size):
                break
            if not is_piece(board, tmp_index[0], tmp_index[1]):
                # No blocking piece
                if tmp_index not in restricted or is_king(board, index[0], index[1]):
                    legal_moves[i] = 1
            else:
                # Blocking piece
                dirty_indices.append((tmp_index[0], tmp_index[1]))
                break
            i += 1
    # Update the cache, dirty map, and dirty flags.
    cache[:, index[0], index[1]] = legal_moves
    dirty_map[index] = dirty_indices
    dirty_flags.remove(index)
    return cache[:, index[0], index[1]]


cpdef update_action_space(board: np.array,
                        cache: np.array,
                        dirty_map: dict,
                        dirty_flags: np.array,
                        ) except *:
    cdef int r
    cdef int c
    for (r, c) in dirty_flags.copy():
        _ = get_moves(board,
                      (r, c),
                      cache,
                      dirty_map,
                      dirty_flags,
                      )


def has_moves(board: np.array,
              cache,
              dirty_map,
              dirty_flags,
              player: str = 'defenders'):
    """Check whether a player has any legal moves."""
    update_action_space(board=board, cache=cache, dirty_map=dirty_map, dirty_flags=dirty_flags)
    if player == 'attackers':
        mask = board[0, :, :] == 1
    else:
        mask = np.sum(board[1:, :, :], axis=0) == 1
    return cache[:, mask].any()


cpdef all_legal_moves(board: np.array,
                      cache: np.array,
                      dirty_map: dict,
                      dirty_flags: np.array,
                      player: str = 'defenders') except *:
    """Return the action-space of all legal moves for a single player"""
    update_action_space(board=board, cache=cache, dirty_map=dirty_map, dirty_flags=dirty_flags)
    if player == 'attackers':
        mask = board[0, :, :] != 1
    else:
        mask = np.sum(board[1:, :, :], axis=0) != 1
    action_space = np.array(cache)
    action_space[:, mask] = 0
    return action_space


def make_move(board: np.array,
              index: Tuple[int, int],
              move: np.int64,
              cache: np.array,
              dirty_map: dict,
              dirty_flags: np.array,
              ) -> tuple:
    """Move the piece at index according to move. Assumes the move is legal."""
    # Find which plane the piece is on (which piece type it is)
    cdef int plane
    if board[0, index[0], index[1]] == 1:
        plane = 0
    elif board[1, index[0], index[1]] == 1:
        plane = 1
    else:
        plane = 2

    # Get the move axis, direction, and number of tiles
    axis = 0 if move < 20 else 1
    direction = 1 if move >= 30 or (20 > move >= 10) else -1
    num = (move % 10) + 1
    # Move the move to the new index and set the old index to 0
    # Consider changing this from axis, direction to row, col to avoid casting
    new_index = list(index)
    new_index[axis] += direction * num
    new_index = tuple(new_index)
    board[plane, new_index[0], new_index[1]] = 1
    board[plane, index[0], index[1]] = 0

    # Due to the move, we need to invalidate the cache of old and new location
    dirty_flags.add(index)
    dirty_flags.add(new_index)

    # Update the cache of the indices affected from the move from old location
    for affected_index in dirty_map[index]:
        dirty_flags.add(affected_index)

    # Now update the cache and dirty map at the new location
    _ = get_moves(board,
                  new_index,
                  cache,
                  dirty_map,
                  dirty_flags,
                  )
    for affected_index in dirty_map[new_index]:
        dirty_flags.add(affected_index)

    return new_index


def get_nice_variables(board: np.array,
                       index: tuple,
                       ) -> tuple:
    """Return some convenient variables used in multiple game utilities"""
    row, col = index
    size = board.shape[-1] - 1
    hostile = [(0, 0), (0, size), (size, 0), (size, size)]
    plane = np.argwhere(board[:, index[0], index[1]] == 1).item()
    ally = TEAMS[plane]
    return row, col, TEAMS, size, hostile, plane, ally


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

    # All of these if statements could probably be collapsed in a similar way as check_shield_wall()
    if row > 0 and is_enemy(row - 1, col):
        if is_edge(row - 1, col, size):
            tags = []
            if check_shield_wall(board, (row - 1, col), tags):
                capture_tags(board, tags)
        # if the enemy is not on an edge, and the other side is an allied piece or hostile piece
        if row - 2 >= 0 and is_flanked(row - 2, col):
            # Destroy it!
            board[:, row - 1, col] = 0

    if row < size and is_enemy(row + 1, col):
        if is_edge(row + 1, col, size):
            tags = []
            if check_shield_wall(board, (row + 1, col), tags):
                capture_tags(board, tags)
        if row + 2 <= size and is_flanked(row + 2, col):
            board[:, row + 1, col] = 0

    if col > 0 and is_enemy(row, col - 1):
        if is_edge(row, col - 1, size):
            tags = []
            if check_shield_wall(board, (row, col - 1), tags):
                capture_tags(board, tags)
        if col - 2 >= 0 and is_flanked(row, col - 2):
            board[:, row, col - 1] = 0

    if col < size and is_enemy(row, col + 1):
        if is_edge(row, col + 1, size):
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
                      edge: str = '',
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
        if is_ally(board, b_dirs[0][0], b_dirs[0][1], ally) and (b_dirs[0][0], b_dirs[0][1]) not in tags:
            adjacent_friends.append((b_dirs[0][0], b_dirs[0][1]))
        if is_ally(board, b_dirs[1][0], b_dirs[1][1], ally) and (b_dirs[1][0], b_dirs[1][1]) not in tags:
            adjacent_friends.append((b_dirs[1][0], b_dirs[1][1]))
        for ally in adjacent_friends:
            if not check_shield_wall(board, ally, tags, edge):
                return False
    else:
        return False
    return True


def is_fort(board, index, defender_tags, interior_tags):
    """Check whether the King is in an edge fort."""
    # This currently doesn't consider corners to be legal pieces of a fort, but it should.
    row, col = index
    size = board.shape[-1] - 1
    interior_tags.append(index)
    adjacent_interior = []
    for step in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        if not in_bounds(row + step[0], col + step[1], size):
            continue
        elif is_blank(board, row + step[0], col + step[1]) or is_king(board, row + step[0], col + step[1]):
            if (row + step[0], col + step[1]) not in interior_tags:
                adjacent_interior.append((row + step[0], col + step[1]))
        elif is_defender(board, row + step[0], col + step[1]):
            if (row + step[0], col + step[1]) not in defender_tags:
                defender_tags.append((row + step[0], col + step[1]))
        else:
            return False

    for tile in adjacent_interior:
        if not is_fort(board, tile, defender_tags, interior_tags):
            return False
    return True


def verify_encirclement(board):
    size = board.shape[-1] - 1
    queue = deque()
    visited = set()
    attacker_walls = []
    interior_tiles = []
    start_row, start_col = find_king(board)
    queue.append((start_row, start_col))
    visited.add((start_row, start_col))
    interior_tiles.append((start_row, start_col))

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        row, col = queue.popleft()
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if in_bounds(nr, nc, size) and (nr, nc) not in visited:
                visited.add((nr, nc))
                if is_defender(board, nr, nc) or is_blank(board, nr, nc):
                    interior_tiles.append((nr, nc))
                    queue.append((nr, nc))
                elif is_attacker(board, nr, nc):
                    attacker_walls.append((nr, nc))

    return attacker_walls, interior_tiles


def is_impenetrable(board, wall_tags, interior_tags, option='fort'):
    size = board.shape[-1] - 1
    if option == 'encirclement':
        is_wall = is_attacker
        is_safe = lambda r, c: (r, c) not in interior_tags
    else:
        is_wall = is_defender
        is_safe = lambda r, c: (r, c) in interior_tags

    def vertical_vuln(r, c):
        if ((not in_bounds(r - 1, c, size) or is_wall(board, r - 1, c) or is_safe(r - 1, c)) or
                (not in_bounds(r + 1, c, size) or is_wall(board, r + 1, c) or is_safe(r + 1, c))):
            return False
        else:
            return True

    def horizontal_vuln(r, c):
        if ((not in_bounds(r, c - 1, size) or is_wall(board, r, c - 1) or is_safe(r, c - 1)) or
                (not in_bounds(r, c + 1, size) or is_wall(board, r, c + 1) or is_safe(r, c + 1))):
            return False
        else:
            return True

    for wall in wall_tags:
        row, col = wall
        if vertical_vuln(row, col) or horizontal_vuln(row, col):
            return False

    return True


def check_encirclement(board):
    size = board.shape[-1] - 1
    queue = deque()
    visited = set()

    for i in range(size + 1):
        queue.append((i, 0))
        queue.append((0, i))
        queue.append((i, size))
        queue.append((size, i))
        visited.add((i, 0))
        visited.add((0, i))
        visited.add((i, size))
        visited.add((size, i))

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        row, col = queue.popleft()
        if is_attacker(board, row, col):
            continue
        for dr, dc in directions:
            nr, nc = row + dr, col + dc

            if in_bounds(nr, nc, size) and (nr, nc) not in visited:
                visited.add((nr, nc))
                if is_defender(board, nr, nc) or is_king(board, nr, nc):
                    return False
                elif is_blank(board, nr, nc):
                    queue.append((nr, nc))

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
    is_hostile = lambda row, col: (row, col) == throne or (
            board_array[:, row, col].any() and TEAMS[np.argwhere(board_array[:, row, col] == 1).item()] != 2)

    # Has the King escaped?
    if (row, col) in corners:
        return 1

    # Is the king surrounded?
    if ((row - 1 > 0 and is_hostile(row - 1, col)) and
            (row + 1 <= size and is_hostile(row + 1, col)) and
            (col - 1 > 0 and is_hostile(row, col - 1)) and
            (col + 1 <= size and is_hostile(row, col + 1))
    ):
        return -1
    return 0


def is_terminal(board,
                cache,
                dirty_map,
                dirty_flags,
                player,
                attacker_moves=1,
                defender_moves=1,
                ):
    king_state = check_king(board)
    if king_state == 1:
        #print("King escaped.")
        return "defenders"
    elif king_state == -1:
        #print("King captured.")
        return "attackers"
    elif player == "defenders":
        king_r, king_c = find_king(board)
        defender_tags = []
        interior_tags = []
        if (is_edge(king_r, king_c, board.shape[-1] - 1) and
           is_fort(board, (king_r, king_c), defender_tags, interior_tags) and
           is_impenetrable(board, defender_tags, interior_tags)):
            #print("Defenders have built an Exit Fort.")
            return "defenders"
    elif player == "attackers":
        if check_encirclement(board):
            attacker_walls, visited = verify_encirclement(board)
            if is_impenetrable(board, attacker_walls, visited, option='encirclement'):
                #print("Attackers have formed an encirclement.")
                return "attackers"
    # Players already check their legal moves each turn, so this is a redundant (and expensive) operation.
    # Optimize this by just passing in the known number of legal moves. If it's zero, that player loses.
    if defender_moves < 10 and not has_moves(board=board, cache=cache, dirty_map=dirty_map,
                                             dirty_flags=dirty_flags, player="defenders"):
        #print("The defenders have no legal moves.")
        return "attackers"
    elif attacker_moves < 10 and not has_moves(board=board, cache=cache, dirty_map=dirty_map,
                                               dirty_flags=dirty_flags, player="attackers"):
        #print("The attackers have no legal moves.")
        return "defenders"

    # Does not check for draws
    else:
        return None
