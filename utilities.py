from collections import deque
# import cython
import numpy as np
# cimport numpy as np
from typing import List, Tuple

# This is a global constant that maps piece planes to team. It is game specific.
TEAMS = {0: 1, 1: 2, 2: 2}
# This global variable is set with set_global_piece_flags during the initialization of the game.
# It is read by is_piece and can be written to by make_move(), check_capture(), and capture_tags().
PIECE_FLAGS = np.array([])


def set_global_piece_flags(piece_flags: np.array) -> None:
    """
    Set the global variable PIECE_FLAGS.
    This should only be called once at the start of the game!

    :param np.array piece_flags: A 2D binary array where 1 means a piece is present at (i, j), 0 means not present.
    """
    global PIECE_FLAGS
    PIECE_FLAGS = piece_flags


# Several small convenience functions that are used in multiple places for condition checks
def is_piece(row: int, col: int) -> int:
    """
    Return 1 if a piece is located at (row, col), 0 otherwise.

    :param int row: The row index.
    :param int col: The col index.
    :return: 1 if there is a piece at (row, col), 0 otherwise.
    """
    # This could cause an index out of bounds error!
    return PIECE_FLAGS[row, col]


def is_king(board: np.array, row: int, col: int) -> bool:
    """
    Return True if the King is located at (row, col), False otherwise.

    :param np.array board: The 3D NumPy array "board" on which the game is being played.
    :param int row: The row index.
    :param int col: The col index.
    :return: True if the King is at (row, col), False otherwise
    """
    return board[2, row, col] == 1


def find_king(board: np.array) -> Tuple[int, int]:
    """
    Returns the current location of the King.

    :param np.array board: The 3D NumPy array "board" on which the game is being played.
    :return: The (row, col) tuple location of the King.
    """
    return tuple(np.argwhere(board[2, :, :] == 1)[0])


def is_defender(board: np.array, row: int, col: int) -> bool:
    """
    Return True if a defender is located at (row, col), False otherwise.

    :param np.array board: The 3D NumPy array "board" on which the game is being played.
    :param int row: The row index.
    :param int col: The col index.
    :return: True if a defender is at (row, col), False otherwise
    """
    return board[1, row, col] == 1


def is_attacker(board: np.array, row: int, col: int) -> bool:
    """
    Return True if an attacker is located at (row, col), False otherwise.

    :param np.array board: The 3D NumPy array "board" on which the game is being played.
    :param int row: The row index.
    :param int col: The col index.
    :return: True if an attacker is at (row, col), False otherwise
    """
    return board[0, row, col] == 1


def is_edge(row: int, col: int, size: int) -> bool:
    """
    Return True if (row, col) is on the edge of the board, False otherwise.

    :param int row: The row index.
    :param int col: The col index.
    :param int size: The size of the board. This is the max valid index. Expected to be len(board) - 1, not len(board).
    :return: True if (row, col) is on the edge of the board, False otherwise
    """
    return row == 0 or col == 0 or row == size or col == size


def is_corner(row: int, col: int, size: int) -> bool:
    """
    Return True if (row, col) is a corner, False otherwise.

    :param int row: The row index.
    :param int col: The col index.
    :param int size: The size of the board. This is the max valid index. Expected to be len(board) - 1, not len(board).
    :return: True if (row, col) is a corner of the board, False otherwise
    """
    return (row, col) == (0, 0) or (row, col) == (0, size) or (row, col) == (size, 0) or (row, col) == (size, size)


def in_bounds(row: int, col: int, size: int) -> bool:
    """
    Return True if (row, col) is a legal index, False otherwise.

    This is used extensively to short-circuit chained conditional statements and
    break out of loops if the board end is reached.

    :param int row: The row index.
    :param int col: The col index.
    :param int size: The size of the board. This is the max valid index. Expected to be len(board) - 1, not len(board).
    :return: True if (row, col) is on the board (a legal index), False otherwise
    """
    return 0 <= row <= size and 0 <= col <= size


def is_blank(row: int, col: int, size: int) -> bool:
    """
    Return True if NO piece is located at (row, col), False otherwise.

    :param int row: The row index.
    :param int col: The col index.
    :param int size: The size of the board. This is the max valid index. Expected to be len(board) - 1, not len(board).
    :return: True if there is NO piece at (row, col), False otherwise.
    """
    return in_bounds(row, col, size) and not is_piece(row, col)


def near_blank(row: int, col: int, size: int) -> bool:
    """
    Return True if at least one tile adjacent to (row, col) is blank, False otherwise.

    :param int row: The row index.
    :param int col: The col index.
    :param int size: The size of the board. This is the max valid index. Expected to be len(board) - 1, not len(board).
    :return:
    """
    return (is_blank(row - 1, col, size) or is_blank(row + 1, col, size) or
            is_blank(row, col - 1, size) or is_blank(row, col + 1, size))


def is_ally(board: np.array, row: int, col: int, ally: int) -> bool:
    """
    Returns True if a piece at (row, col) is the same team as the arg ally.

    :param np.array board: The 3D NumPy array "board" on which the game is being played.
    :param int row: The row index.
    :param int col: The col index.
    :param int ally: The team being checked against.
    :return: True if a piece at (row, col) is on the team of the arg "ally".
    """
    return is_piece(row, col) and TEAMS[np.argwhere(board[:, row, col] == 1).item()] == ally


def is_enemy(board: np.array, row: int, col: int, plane: int) -> bool:
    """
    Return True if the piece at (row, col) is a non-King enemy according plane.

    :param np.array board: The 3D NumPy array "board" on which the game is being played.
    :param int row: The row index.
    :param int col: The col index.
    :param int plane: The piece plane. is_enemy checks for non-King "enemies" from the perspective of this plane.
    :return: True if the (row, col) contains a non-King enemy, False otherwise.
    """
    return is_piece(row, col) and np.argwhere(board[:, row, col] == 1).item() not in [plane, 2]


def is_hostile(board: np.array, row: int, col: int, ally: int, hostile: List[tuple,]) -> bool:
    """
    Returns True if (row, col) is hostile to the "ally" team.

    :param np.array board: The 3D NumPy array "board" on which the game is being played.
    :param int row:  The row index.
    :param int col:  The col index.
    :param int ally:  The team being checked against.
    :param list hostile: A list of hostile tiles, such as corners or (sometimes) the throne.
    :return: True if (row, col) is hostile to the "ally" team, False otherwise.
    """
    return ((is_piece(row, col) and TEAMS[np.argwhere(board[:, row, col] == 1).item()] != ally) or
            (row, col) in hostile)


def is_flanked(board: np.array, row: int, col: int, ally: int, hostile: List[tuple,]) -> bool:
    """
    Returns True if (row, col) is flanked by hostile tiles.

    :param np.array board: The 3D NumPy array "board" on which the game is being played.
    :param int row:  The row index.
    :param int col:  The col index.
    :param int ally:  The team being checked against.
    :param list hostile: A list of hostile tiles, such as corners or (sometimes) the throne.
    :return: True if (row, col) flanked by hostile tiles, False otherwise.
    """
    return ((is_piece(row, col) and TEAMS[np.argwhere(board[:, row, col] == 1).item()] == ally) or
            ((row, col) in hostile))


def get_nice_variables(board: np.array,
                       index: tuple,
                       ) -> tuple:
    """
    Return some convenient variables used in multiple game utilities

    :param np.array board: The 3D NumPy array "board" on which the game is being played.e
    :param Tuple[int, int] index: The index of the relevant piece that we're interested in.
    :return: A tuple containing several variables of use.
    """
    row, col = index
    size = board.shape[1] - 1
    hostile = {(0, 0), (0, size), (size, 0), (size, size)}
    if board[0, index[0], index[1]] == 1:
        plane = 0
    elif board[1, index[0], index[1]] == 1:
        plane = 1
    else:
        plane = 2
    ally = TEAMS[plane]
    return row, col, TEAMS, size, hostile, plane, ally


def quiescent_defender(board: np.array, cache: np.array, dirty_map: dict, dirty_flags: set) -> bool:
    """
    Returns True if the Defenders have imminent victory (they can win if they move right now), False otherwise.

    This function is designed as a minimalist heuristic to guide MCTS rollouts by keeping the rollouts close to random,
    but deterministically terminating them if a state is reached in which a real game would surely end.
    These semi-random rollouts typically take 50% - 70% fewer turns than a genuinely random rollout would.

    :param np.array board: The 3D NumPy array "board" on which the game is being played.
    :param np.array cache: The 3D NumPy array cache of moves.
    :param dirty_map: A dictionary mapping index value i to a list of indices j that would experience cache invalidation
                      if i moves, e.g. if i moves, the legal moves for every j need to be refreshed.
    :param dirty_flags: A set of tuples that need to have their legal move cache refreshed.
    :return: True if the Defenders can immediately win by the King escaping, False otherwise.
    """
    row, col = find_king(board)
    size = board.shape[2] - 1
    king_moves = get_moves(board=board, index=(row, col), cache=cache,
                           dirty_map=dirty_map, dirty_flags=dirty_flags)

    # If the King is on any edge, then there are exactly two moves that he may have that would end the game.
    # Depending on which edge he is on and where, we check those two moves. If at least one is legal,
    # the Defenders have imminent victory. Return True.
    if col == 0 or col == size:
        if king_moves[row - 1] == 1 or king_moves[9 + (size - row)]:
            return True
    elif row == 0 or row == size:
        if king_moves[19 + col] == 1 or king_moves[29 + (size - col)] == 1:
            return True
    else:
        return False


def quiescent_attacker(board: np.array) -> bool:
    """
    Returns True if the Attackers have imminent victory (they can win if they move right now), False otherwise.

    This function is designed as a minimalist heuristic to guide MCTS rollouts by keeping the rollouts close to random,
    but deterministically terminating them if a state is reached in which a real game would surely end.
    These semi-random rollouts typically take 50% - 70% fewer turns than a genuinely random rollout would.

    :param np.array board: The 3D NumPy array "board" on which the game is being played.
    :return: True if the Attackers can immediately win by capturing the King, False otherwise.
    """
    row, col = find_king(board)
    num_surrounding = 0
    size = board.shape[2] - 1
    sides = {(row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)}

    # Check the four tiles adjacent to the King, who is at (row, col)
    if in_bounds(row + 1, col, size) and (is_attacker(board, row + 1, col) or (row + 1, col) == (size//2, size/2)):
        num_surrounding += 1
        sides.remove((row + 1, col))
    if in_bounds(row - 1, col, size) and (is_attacker(board, row - 1, col) or (row - 1, col) == (size//2, size/2)):
        num_surrounding += 1
        sides.remove((row - 1, col))
    if in_bounds(row, col + 1, size) and (is_attacker(board, row, col + 1) or (row, col + 1) == (size//2, size/2)):
        num_surrounding += 1
        sides.remove((row, col + 1))
    if in_bounds(row, col - 1, size) and (is_attacker(board, row, col - 1) or (row, col - 1) == (size//2, size/2)):
        num_surrounding += 1
        sides.remove((row, col - 1))

    # If the King already has three hostile tiles around him, examine the fourth blank tile.
    if num_surrounding == 3:
        open_space = sides.pop()
        if is_blank(open_space[0], open_space[1], size):
            for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                tr, tc = open_space
                # If we can walk from the blank tile in any direction and bump an attacker, that attacker could
                # capture the King. Therefore, the Attackers have imminent victory, return True.
                while True:
                    tr += dr
                    tc += dc
                    if in_bounds(tr, tc, size) and not is_piece(tr, tc):
                        continue
                    elif in_bounds(tr, tc, size) and is_attacker(board, tr, tc):
                        return True
                    else:
                        break
        return False


# The group of small convenience functions ends here


def get_moves(board: np.array,
              index: Tuple[int, int],
              cache: np.array,
              dirty_map: dict,
              dirty_flags: set,
              ) -> np.array:
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

    # If there is no piece on this tile, return immediately (there are no legal moves if there's no piece!)
    if not is_piece(index[0], index[1]):
        return np.zeros(40)

    # If the cache of this index is not dirty, immediately return the cached legal moves.
    elif index not in dirty_flags:
        return cache[:, index[0], index[1]]
    #cdef int size
    size = board.shape[2] - 1
    restricted = [[0, 0], [0, size], [size, 0], [size, size], [size // 2, size // 2]]
    dirty_indices = []
    legal_moves = np.zeros(40)
    # Need to go through the 40 possible moves and check the legality of each...
    # 0-9 is up 1-10, 10-19 is down 1-10, 20-29 is left 1-10, 30-39 is right 1-10

    #cdef int k
    #cdef tuple instruction
    #cdef int i
    #cdef int axis
    #cdef int direction
    #cdef int dx[4]
    dx = [0, 0, 1, 1]
    #dx[0], dx[1], dx[2], dx[3] = 0, 0, 1, 1
    #cdef int dy[4]
    dy = [-1, 1, -1, 1]
    #dy[0], dy[1], dy[2], dy[3] = -1, 1, -1, 1
    #cdef int initial_row
    #cdef int initial_col
    initial_row = index[0]
    initial_col = index[1]
    #cdef int tmp_index[2]
    tmp_index = [initial_row, initial_col]

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
            if not is_piece(tmp_index[0], tmp_index[1]):
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


def update_action_space(board: np.array,
                        cache: np.array,
                        dirty_map: dict,
                        dirty_flags: set,
                        ) -> None:
    """
    Refresh any dirty cache locations.

    :param np.array board: The 3D NumPy array "board" on which the game is being played.
    :param np.array cache: The 3D NumPy array cache of moves.
    :param dirty_map: A dictionary mapping index value i to a list of indices j that would experience cache invalidation
                      if i moves, e.g. if i moves, the legal moves for every j need to be refreshed.
    :param dirty_flags: A set of tuples that need to have their legal move cache refreshed.
    :return: None
    """

    # This loop will modify the dirty_flags set since it's refreshing the cache.
    # We need a copy of the set to avoid iterating over a structure that we're modifying.
    for (r, c) in dirty_flags.copy():
        _ = get_moves(board,
                      (r, c),
                      cache,
                      dirty_map,
                      dirty_flags,
                      )


def has_moves(board: np.array,
              cache: np.array,
              dirty_map: dict,
              dirty_flags: set,
              player: str = 'defenders',
              ) -> bool:
    """
    Check whether a player has any legal moves.

    This should only be called if we know that a player has so few moves that
    their opponent may have eliminated their last legal move in the last turn. The program already passively already
    tracks the number of legal moves that a player has, so this function is basically a way to explicitly confirm
    that a player has no moves prior to terminating the game.

    :param np.array board: The 3D NumPy array "board" on which the game is being played.
    :param np.array cache: The 3D NumPy array cache of moves.
    :param dirty_map: A dictionary mapping index value i to a list of indices j that would experience cache invalidation
                      if i moves, e.g. if i moves, the legal moves for every j need to be refreshed.
    :param dirty_flags: A set of tuples that need to have their legal move cache refreshed.
    :param str player: The player. It is either "attackers" or "defenders".
    :return: True if player has remaining moves, False if player has no legal moves.
    """
    update_action_space(board=board, cache=cache, dirty_map=dirty_map, dirty_flags=dirty_flags)
    if player == 'attackers':
        mask = board[0, :, :] == 1
    else:
        mask = np.sum(board[1:, :, :], axis=0) == 1
    return cache[:, mask].any()


def all_legal_moves(board: np.array,
                    cache: np.array,
                    dirty_map: dict,
                    dirty_flags: set,
                    player: str = 'defenders',
                    ) -> np.array:
    """
    Return a 3D NumPy array representing the action-space of a single player.

    :param np.array board: The 3D NumPy array "board" on which the game is being played.
    :param np.array cache: The 3D NumPy array cache of moves.
    :param dirty_map: A dictionary mapping index value i to a list of indices j that would experience cache invalidation
                      if i moves, e.g. if i moves, the legal moves for every j need to be refreshed.
    :param dirty_flags: A set of tuples that need to have their legal move cache refreshed.
    :param str player: The player. It is either "attackers" or "defenders".
    :return: A 3D NumPy array representing the action-space of a single player. For standard hnefatafl, it is 40x11x11.
    """
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
              move: int,
              cache: np.array,
              dirty_map: dict,
              dirty_flags: set,
              ) -> tuple:
    """
    Move the piece at index according to move. Assumes the move is legal.

    :param np.array board: The 3D NumPy array "board" on which the game is being played.
    :param Tuple[int, int] index: The index of the piece to be moved.
    :param int move: The encoded move that the piece at index will make.
    :param np.array cache: The 3D NumPy array cache of moves.
    :param dirty_map: A dictionary mapping index value i to a list of indices j that would experience cache invalidation
                      if i moves, e.g. if i moves, the legal moves for every j need to be refreshed.
    :param dirty_flags: A set of tuples that need to have their legal move cache refreshed.
    :return:
    """

    # Find which plane the piece is on (which piece type it is)
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

    # Move the piece to the new index and set the old index to 0
    new_index = list(index)
    new_index[axis] += direction * num
    new_index = tuple(new_index)
    board[plane, new_index[0], new_index[1]] = 1
    board[plane, index[0], index[1]] = 0

    # Update PIECE_FLAGS. This is writing to a global variable.
    PIECE_FLAGS[index[0], index[1]] = 0
    PIECE_FLAGS[new_index[0], new_index[1]] = 1

    # Due to the move, we need to invalidate the cache of old and new location
    dirty_flags.add(index)
    dirty_flags.add(new_index)

    # Update the cache of the indices affected by the move from the old location
    for affected_index in dirty_map[index]:
        dirty_flags.add(affected_index)

    # Now update the cache and dirty map at the new location
    # Note that the reason we're doing this here is that we refresh the dirty_map as a byproduct of
    # get_moves(), and we need that to be able to set the dirty flags at the new location.
    _ = get_moves(board,
                  new_index,
                  cache,
                  dirty_map,
                  dirty_flags,
                  )

    # Update the cache of the indices affected by the move to the new location
    for affected_index in dirty_map[new_index]:
        dirty_flags.add(affected_index)

    return new_index


def check_capture(board: np.array,
                  index: tuple,
                  ) -> None:
    """
    Given an index, checks to see if any basic enemies pieces around it are captured.

    :param np.array board: The 3D NumPy array "board" on which the game is being played.
    :param Tuple[int, int] index: The index of the piece around which we check for pieces to capture.
    :return: None; if a piece can be captured, it will be automatically within this function.
    """
    # Set up some convenient variables
    row, col, teams, size, hostile, plane, ally = get_nice_variables(board, index)

    # If the throne is empty, it is hostile
    if not board[:, size // 2, size // 2].any():
        hostile.add((size // 2, size // 2))

    # All of these if statements could probably be collapsed in a similar way as check_shield_wall()
    if row > 0 and is_enemy(board, row - 1, col, plane):
        if is_edge(row - 1, col, size):
            tags = []
            if check_shield_wall(board, (row - 1, col), tags):
                capture_tags(board, tags)
        # if the enemy is not on an edge, and the other side is an allied piece or hostile piece
        if row - 2 >= 0 and is_flanked(board, row - 2, col, ally, hostile):
            # Destroy it!
            board[:, row - 1, col] = 0
            PIECE_FLAGS[row - 1, col] = 0

    if row < size and is_enemy(board, row + 1, col, plane):
        if is_edge(row + 1, col, size):
            tags = []
            if check_shield_wall(board, (row + 1, col), tags):
                capture_tags(board, tags)
        if row + 2 <= size and is_flanked(board, row + 2, col, ally, hostile):
            board[:, row + 1, col] = 0
            PIECE_FLAGS[row + 1, col] = 0

    if col > 0 and is_enemy(board, row, col - 1, plane):
        if is_edge(row, col - 1, size):
            tags = []
            if check_shield_wall(board, (row, col - 1), tags):
                capture_tags(board, tags)
        if col - 2 >= 0 and is_flanked(board, row, col - 2, ally, hostile):
            board[:, row, col - 1] = 0
            PIECE_FLAGS[row, col - 1] = 0

    if col < size and is_enemy(board, row, col + 1, plane):
        if is_edge(row, col + 1, size):
            tags = []
            if check_shield_wall(board, (row, col + 1), tags):
                capture_tags(board, tags)
        if col + 2 <= size and is_flanked(board, row, col + 2, ally, hostile):
            board[:, row, col + 1] = 0
            PIECE_FLAGS[row, col + 1] = 0


def capture_tags(board: np.array,
                 tags: list,
                 ) -> None:
    """
    Capture any non-King pieces who are "tagged" as being trapped in a shield wall.

    :param np.array board: The 3D NumPy array "board" on which the game is being played.
    :param tags: A list of tuples. Each tuple is a piece that was tagged as being trapped in a shield wall.
    :return: None; any enemies who are trapped in a shield wall are eliminated by this function.
    """
    for tag in tags:
        if np.argwhere(board[:, tag[0], tag[1]] == 1).item() != 2:
            board[:, tag[0], tag[1]] = 0
            PIECE_FLAGS[tag[0], tag[1]] = 0


def check_shield_wall(board: np.array,
                      index: Tuple[int, int],
                      tags: list,
                      edge: str = '',
                      ) -> bool:
    """
    Recursively check whether a shield wall capture can be executed.

    This is an ugly function and should be rewritten from scratch using queue-based flood-fill, not recursion.

    :param np.array board: The 3D NumPy array "board" on which the game is being played.
    :param Tuple[int, int] index: The index of the piece around which we check for pieces to capture.
    :param tags: A list of tuples. Each tuple is a piece that was tagged as being trapped in a shield wall.
    :param edge: The board edge against which there may be a shield wall.
    :return: True if pieces have been trapped in a shield wall, False otherwise.
    """
    row, col, teams, size, hostile, plane, ally = get_nice_variables(board, index)

    # A shield wall can only happen if units are "pinned" against an edge.
    # To check if they're pinned, we need to know the edge along which we're checking.
    if not edge:
        if row == 0:
            edge = 'up'
        elif row == size:
            edge = 'down'
        elif col == 0:
            edge = 'left'
        else:
            edge = 'right'

    h_mapping = {'up': (row + 1, col), 'down': (row - 1, col), 'left': (row, col + 1), 'right': (row, col - 1)}
    b_mapping = {'up': ((row, col - 1), (row, col + 1)), 'down': ((row, col - 1), (row, col + 1)),
                 'left': ((row - 1, col), (row + 1, col)), 'right': ((row - 1, col), (row + 1, col))}
    h_dir = h_mapping[edge]
    b_dirs = b_mapping[edge]

    if board[0, h_dir[0], h_dir[1]] == 1:
        plane = 0
    elif board[1, h_dir[0], h_dir[1]] == 1:
        plane = 1
    else:
        plane = 2

    if (
       ((is_piece(h_dir[0], h_dir[1]) and TEAMS[plane] != ally) or (h_dir[0], h_dir[1]) in hostile) and
       (is_piece(b_dirs[0][0], b_dirs[0][1]) or (b_dirs[0][0], b_dirs[0][1]) in hostile) and
       (is_piece(b_dirs[1][0], b_dirs[1][1]) or (b_dirs[1][0], b_dirs[1][1]) in hostile)
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


def is_fort(board: np.array,
            index: Tuple[int, int],
            defender_tags: List[Tuple[int, int],],
            interior_tags: List[Tuple[int, int],],
            ) -> bool:
    """
    Check whether the King is in an edge fort.

    :param np.array board: The 3D NumPy "board" array on which the game is being played.
    :param Tuple[int, int] index: The current index of the tile being checked with fort logic.
    :param list defender_tags: A list of defender indices. If this is a fort, then these are the "walls" of the fort.
    :param list interior_tags: A list of interior tiles. If this is a fort, then these are "inside" the fort.
    :return: True if a fort has been made, False otherwise. This does not guarantee that the fort is impenetrable.
    """
    row, col = index
    size = board.shape[1] - 1
    interior_tags.append(index)
    adjacent_interior = []

    # Check in each of the 4 directions around index.
    # If it's out of bounds, ignore it. If it's a blank that hasn't been checked yet, add it to adjacent_interior.
    # If it's a defender that hasn't been tagged yet, add it to defender_tags.
    # Otherwise, it must be an attacker or a corner, so this can't be a fort. Return false.
    for step in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        if not in_bounds(row + step[0], col + step[1], size):
            continue
        elif is_blank(row + step[0], col + step[1], size) or is_king(board, row + step[0], col + step[1]):
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


def verify_encirclement(board: np.array) -> Tuple[list, list]:
    """
    Return the "walls" of an encirclement.

    The function check_encirclement() doesn't record the attackers who form the encirclement, because almost always
    there won't be an encirclement. So it chooses to quickly try to dismiss encirclement and not contrive to figure out
    attackers who may form an encirclement. If it turns out that there is an encirclement, this algorithm efficiently
    uses a queue-based flood fill from the defenders outward to find the attackers who make up the "wall" around them.
    Then we use is_impenetrable with option='encirclement' to confirm that it is a game-ending encirclement that
    can't be escaped.

    :param np.array board: The 3D NumPy "board" array on which the game is being played.
    :return: A tuple containing two lists, one list of the encircling attackers and one list of encircled tiles.
    """

    size = board.shape[1] - 1
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
                if is_defender(board, nr, nc) or is_blank(nr, nc, size):
                    interior_tiles.append((nr, nc))
                    queue.append((nr, nc))
                elif is_attacker(board, nr, nc):
                    attacker_walls.append((nr, nc))

    return attacker_walls, interior_tiles


def is_impenetrable(board: np.array,
                    wall_tags: list,
                    interior_tags: list,
                    option: str = 'fort',
                    ) -> bool:
    """
    Confirm whether a fort/encirclement is impenetrable (game-ending).

    :param board: The 3D NumPy 'board' array on which the game is being played.
    :param wall_tags: A list of (row, col) tuples, each of which is a wall of the fort/encirclement.
    :param interior_tags: A list of (row, col) tuples, each of which is a tile inside the fort/encirclement.
    :param option: 'fort' or 'encirclement', depending on what we are checking for impenetrability.
    :return: True if the fort/encirclement is impenetrable, False otherwise.
    """

    size = board.shape[1] - 1
    if option == 'encirclement':
        is_wall = is_attacker
        is_safe = lambda r, c: (r, c) not in interior_tags
    else:
        is_wall = is_defender
        is_safe = lambda r, c: (r, c) in interior_tags

    def vertical_vulnerable(r, c):
        if ((not in_bounds(r - 1, c, size) or is_wall(board, r - 1, c) or is_safe(r - 1, c)) or
           (not in_bounds(r + 1, c, size) or is_wall(board, r + 1, c) or is_safe(r + 1, c))):
            return False
        else:
            return True

    def horizontal_vulnerable(r, c):
        if ((not in_bounds(r, c - 1, size) or is_wall(board, r, c - 1) or is_safe(r, c - 1)) or
           (not in_bounds(r, c + 1, size) or is_wall(board, r, c + 1) or is_safe(r, c + 1))):
            return False
        else:
            return True

    for wall in wall_tags:
        row, col = wall
        if vertical_vulnerable(row, col) or horizontal_vulnerable(row, col):
            return False

    return True


def check_encirclement(board: np.array) -> bool:
    """
    Check whether the attackers have encircled all defenders.

    :param np.array board: The 3D NumPy 'board' array on which the game is being played.
    :return: True if the attackers have encircled all defenders, otherwise False.
    """
    size = board.shape[1] - 1
    queue = deque()
    visited = set()

    # This is queueing every tile that is on the edge.
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

    # The algorithm is an outward-in flood-fill. If it touches a defender, defenders are not encircled.
    # If the queue comes up empty never having touched a defender, then the defenders must be encircled.
    while queue:
        row, col = queue.popleft()
        if is_defender(board, row, col):
            return False
        elif is_attacker(board, row, col):
            continue

        for dr, dc in directions:
            nr, nc = row + dr, col + dc

            if in_bounds(nr, nc, size) and (nr, nc) not in visited:
                visited.add((nr, nc))
                if is_defender(board, nr, nc) or is_king(board, nr, nc):
                    return False
                elif is_blank(nr, nc, size):
                    queue.append((nr, nc))

    return True


def check_king(board: np.array,
               ) -> int:
    """
    Check whether the King has escaped or been captured.

    :param np.array board: A 2D NumPy array representing the board
    :return: -1 means King captured, 1 means King escaped, 0 means neither.
    """
    size = board.shape[1] - 1
    row, col = find_king(board)
    corners = [(0, 0), (0, size), (size, 0), (size, size)]
    throne = (size // 2, size // 2)

    # Has the King escaped?
    if (row, col) in corners:
        return 1

    # Is the king surrounded?
    if ((row - 1 > 0 and ((row - 1, col) == throne or
                          (is_piece(row - 1, col) and
                           is_attacker(board, row - 1, col)))) and
        (row + 1 <= size and ((row + 1, col) == throne or
                              (is_piece(row + 1, col) and
                               is_attacker(board, row + 1, col)))) and
        (col - 1 > 0 and ((row, col - 1) == throne or
                          (is_piece(row, col - 1) and
                           is_attacker(board, row, col - 1)))) and
        (col + 1 <= size and ((row, col + 1) == throne or
                              (is_piece(row, col + 1) and
                               is_attacker(board, row, col + 1))))
        ):
        return -1
    return 0


def is_terminal(board: np.array,
                cache: np.array,
                dirty_map: dict,
                dirty_flags: set,
                player: str,
                attacker_moves: int = 1,
                defender_moves: int = 1,
                ):
    """
    Check for termination of the game using all pertinent mechanics.

    :param np.array board: The 3D NumPy 'board' on which the game is being played.
    :param np.array cache: The cache of legal moves for the pieces.
    :param dict dirty_map: A dictionary mapping index i to a list of indices that will have invalidated caches if i moves.
    :param set dirty_flags: A set of (row, col) tuples that have invalid caches and need to be refreshed.
    :param str player: Either "defenders" or "attackers".
    :param int attacker_moves: The number of legal moves that the attackers had as of the latest check.
    :param int defender_moves: The number of legal moves that the attackers had as of the latest check.
    :return: "defenders" or "attackers" if either has won, otherwise None.
    """
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
        if (is_edge(king_r, king_c, board.shape[1] - 1) and
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
