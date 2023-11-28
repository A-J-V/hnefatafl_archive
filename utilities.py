from collections import deque
import random
# import cython
import numpy as np
# cimport numpy as np
from typing import List, Tuple

# This is a global constant that maps piece planes to team. It is game specific.
TEAMS = {0: 1, 1: 2, 2: 2}


# Several small convenience functions that are used in multiple places for condition checks
def is_piece(row: int,
             col: int,
             piece_flags: np.array,
             ) -> int:
    """
    Return 1 if a piece is located at (row, col), 0 otherwise.

    :param int row: The row index.
    :param int col: The col index.
    :param np.array piece_flags: A 2D binary NumPy array. If (row, col) is 1, a piece if present, otherwise no piece.
    :return: 1 if there is a piece at (row, col), 0 otherwise.
    """
    # This could cause an index out of bounds error!
    return piece_flags[row, col]


def is_piece_thin(board: np.array,
                  row,
                  col,
                  ) -> int:
    """
    Return 1 if a piece is located at (row, col), 0 otherwise.
    This is a slower version of is_piece() that doesn't require a piece_flag cache. It's used in move evaluations.
    """
    return board[:, row, col].any()


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


def is_blank(row: int, col: int, size: int, piece_flags: np.array) -> bool:
    """
    Return True if NO piece is located at (row, col), False otherwise.

    :param int row: The row index.
    :param int col: The col index.
    :param int size: The size of the board. This is the max valid index. Expected to be len(board) - 1, not len(board).
    :param np.array piece_flags: A 2D binary NumPy array. If (row, col) is 1, a piece if present, otherwise no piece.
    :return: True if there is NO piece at (row, col), False otherwise.
    """
    return in_bounds(row, col, size) and not is_piece(row, col, piece_flags)


def is_blank_thin(board: np.array,
                  row: int,
                  col: int,
                  size: int,
                  ) -> bool:
    """
    Returns True if (row, col) is blank, False otherwise.
    This is a slower version of is_blank() that doesn't require a piece_flag cache. It's used for move evaluations.
    """
    return in_bounds(row, col, size) and not is_piece_thin(board, row, col)


def near_blank(row: int, col: int, size: int, piece_flags: np.array) -> bool:
    """
    Return True if at least one tile adjacent to (row, col) is blank, False otherwise.

    :param int row: The row index.
    :param int col: The col index.
    :param int size: The size of the board. This is the max valid index. Expected to be len(board) - 1, not len(board).
    :param np.array piece_flags: A 2D binary NumPy array. If (row, col) is 1, a piece if present, otherwise no piece.
    :return:
    """
    return (is_blank(row - 1, col, size, piece_flags) or is_blank(row + 1, col, size, piece_flags) or
            is_blank(row, col - 1, size, piece_flags) or is_blank(row, col + 1, size, piece_flags))


def is_ally(board: np.array, row: int, col: int, ally: int, piece_flags: np.array) -> bool:
    """
    Returns True if a piece at (row, col) is the same team as the arg ally.

    :param np.array board: The 3D NumPy array "board" on which the game is being played.
    :param int row: The row index.
    :param int col: The col index.
    :param int ally: The team being checked against.
    :param np.array piece_flags: A 2D binary NumPy array. If (row, col) is 1, a piece if present, otherwise no piece.
    :return: True if a piece at (row, col) is on the team of the arg "ally".
    """
    return is_piece(row, col, piece_flags) and TEAMS[np.argwhere(board[:, row, col] == 1).item()] == ally


def is_enemy(board: np.array, row: int, col: int, ally: int, piece_flags: np.array) -> bool:
    """
    Return True if the piece at (row, col) is a NON-KING enemy according plane.

    :param np.array board: The 3D NumPy array "board" on which the game is being played.
    :param int row: The row index.
    :param int col: The col index.
    :param int ally: The team of the piece checking for enemies.
    :param np.array piece_flags: A 2D binary NumPy array. If (row, col) is 1, a piece if present, otherwise no piece.
    :return: True if the (row, col) contains a non-King enemy, False otherwise.
    """
    if is_piece(row, col, piece_flags):
        target_plane = np.argwhere(board[:, row, col] == 1).item()
    else:
        return False
    return (TEAMS[target_plane] != ally and
            target_plane != 2)


def is_hostile(board: np.array, row: int, col: int, ally: int, hostile: List[tuple,], piece_flags: np.array) -> bool:
    """
    Returns True if (row, col) is hostile to the "ally" team.

    :param np.array board: The 3D NumPy array "board" on which the game is being played.
    :param int row:  The row index.
    :param int col:  The col index.
    :param int ally:  The team being checked against.
    :param list hostile: A list of hostile tiles, such as corners or (sometimes) the throne.
    :param np.array piece_flags: A 2D binary NumPy array. If (row, col) is 1, a piece if present, otherwise no piece.
    :return: True if (row, col) is hostile to the "ally" team, False otherwise.
    """
    return ((is_piece(row, col, piece_flags) and TEAMS[np.argwhere(board[:, row, col] == 1).item()] != ally) or
            (row, col) in hostile)


def is_flanked(board: np.array, row: int, col: int, ally: int, hostile: List[tuple,], piece_flags: np.array) -> bool:
    """
    Returns True if (row, col) is flanked by hostile tiles.

    :param np.array board: The 3D NumPy array "board" on which the game is being played.
    :param int row:  The row index.
    :param int col:  The col index.
    :param int ally:  The team being checked against.
    :param list hostile: A list of hostile tiles, such as corners or (sometimes) the throne.
    :param np.array piece_flags: A 2D binary NumPy array. If (row, col) is 1, a piece if present, otherwise no piece.
    :return: True if (row, col) flanked by hostile tiles, False otherwise.
    """
    return ((is_piece(row, col, piece_flags) and TEAMS[np.argwhere(board[:, row, col] == 1).item()] == ally) or
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
    plane = get_plane(board, index)
    ally = TEAMS[plane]
    return row, col, TEAMS, size, hostile, plane, ally


def get_plane(board: np.array,
              index: tuple,
              ) -> int:
    """
    Return the plane of index.

    :param board: The 3D NumPy "board" array on which the game is being played.
    :param index: The index whose plane we are checking.
    :return: The plane of the index.
    """
    if board[0, index[0], index[1]] == 1:
        plane = 0
    elif board[1, index[0], index[1]] == 1:
        plane = 1
    else:
        plane = 2
    return plane


def is_vulnerable(board: np.array,
                  index: tuple,
                  ) -> int:
    """
    Checks whether the piece at (row, col) could be captured by the enemy next turn.
    This is used mainly in heuristic evaluation.

    :param board:
    :param index:
    :return:
    """
    plane = get_plane(board, index)
    if plane == 2:
        return 0
    size = board.shape[2] - 1

    vulnerable_tiles = []
    # For every direction, check for an enemy.
    for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        row, col = index
        row += direction[0]
        col += direction[1]
        if in_bounds(row, col, size) and plane == 1 and is_attacker(board, row, col):
            row -= 2 * direction[0]
            col -= 2 * direction[1]
            if in_bounds(row, col, size) and is_blank_thin(board, row, col, size):
                vulnerable_tiles.append((row, col))
        elif in_bounds(row, col, size) and plane == 0 and is_defender(board, row, col):
            row -= 2 * direction[0]
            col -= 2 * direction[1]
            if in_bounds(row, col, size) and is_blank_thin(board, row, col, size):
                vulnerable_tiles.append((row, col))

    for tile in vulnerable_tiles:
        # For every direction, check for vulnerability.
        for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            row, col = tile
            while True:
                row += direction[0]
                col += direction[1]
                if not in_bounds(row, col, size):
                    break
                elif plane == 1 and is_attacker(board, row, col):
                    return True
                elif plane == 0 and is_defender(board, row, col):
                    return True
                elif is_piece_thin(board, row, col):
                    break
    return False


# The group of small convenience functions ends here

# Heuristic functions for early termination begin here
def quiescent_defender(board: np.array,
                       cache: np.array,
                       dirty_map: dict,
                       dirty_flags: set,
                       piece_flags: np.array,
                       ) -> bool:
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
    :param np.array piece_flags: A 2D binary NumPy array. If (row, col) is 1, a piece if present, otherwise no piece.
    :return: True if the Defenders can immediately win by the King escaping, False otherwise.
    """
    row, col = find_king(board)
    size = board.shape[2] - 1
    king_moves = get_moves(board=board,
                           index=(row, col),
                           cache=cache,
                           dirty_map=dirty_map,
                           dirty_flags=dirty_flags,
                           piece_flags=piece_flags,
                           )

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


def quiescent_attacker(board: np.array, piece_flags: np.array) -> bool:
    """
    Returns True if the Attackers have imminent victory (they can win if they move right now), False otherwise.

    This function is designed as a minimalist heuristic to guide MCTS rollouts by keeping the rollouts close to random,
    but deterministically terminating them if a state is reached in which a real game would surely end.
    These semi-random rollouts typically take 50% - 70% fewer turns than a genuinely random rollout would.

    :param np.array board: The 3D NumPy array "board" on which the game is being played.
    :param np.array piece_flags: A 2D binary NumPy array. If (row, col) is 1, a piece if present, otherwise no piece.
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
        if is_blank(open_space[0], open_space[1], size, piece_flags):
            for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                tr, tc = open_space
                # If we can walk from the blank tile in any direction and bump an attacker, that attacker could
                # capture the King. Therefore, the Attackers have imminent victory, return True.
                while True:
                    tr += dr
                    tc += dc
                    if in_bounds(tr, tc, size) and not is_piece(tr, tc, piece_flags):
                        continue
                    elif in_bounds(tr, tc, size) and is_attacker(board, tr, tc):
                        return True
                    else:
                        break
        return False\

# Heuristic functions for early termination end here

# Feature engineering functions begin here


def get_attacker_losses(board: np.array,
                        starting_num: int = 24):
    return starting_num - np.sum(board[0, :, :])


def get_defender_losses(board: np.array,
                        starting_num: int = 12):
    return starting_num - np.sum(board[1, :, :])


def get_material_balance(board: np.array,
                         attacker_starting_num: int = 24,
                         defender_starting_num: int = 12,
                         ) -> int:
    """
    Return the material balance from the perspective of the defenders.
    In other words, if the defenders capture an attacker but haven't lost any pieces, material balance is 1.
    If the attackers capture a defender but haven't lost any pieces, material balance is -1.

    :param board:
    :param attacker_starting_num:
    :param defender_starting_num:
    :return:
    """
    attacker_losses = get_attacker_losses(board, attacker_starting_num)
    defender_losses = get_defender_losses(board, defender_starting_num)
    return attacker_losses - defender_losses


def get_king_distance_to_corner(board: np.array) -> int:
    """Return the King's Manhattan distance to the nearest corner.

    :param np.array board: The board on which the game is being played.
    :return: The integer representing the Manhattan distance from the King to the nearest corner.
    """

    king_loc = find_king(board=board)
    size = board.shape[2] - 1
    center = size // 2

    # Check if King is closer to upper or bottom edge. Need to randomly break ties to avoid bias.
    if king_loc[0] < center:
        vertical = 'up'
    elif king_loc[0] > center:
        vertical = 'down'
    else:
        vertical = random.choice(['up', 'down'])

    # Check if King is closer to left or right edge. Need to randomly break ties to avoid bias.
    if king_loc[1] < center:
        horizontal = 'left'
    elif king_loc[1] > center:
        horizontal = 'right'
    else:
        horizontal = random.choice(['left', 'right'])

    # Return manhattan distance to nearest edge to the King.
    if vertical == 'up' and horizontal == 'right':
        return abs(king_loc[0] - 0) + abs(king_loc[1] - size)
    elif vertical == 'up' and horizontal == 'left':
        return abs(king_loc[0] - 0) + abs(king_loc[1] - 0)
    elif vertical == 'down' and horizontal == 'right':
        return abs(king_loc[0] - size) + abs(king_loc[1] - size)
    elif vertical == 'down' and horizontal == 'left':
        return abs(king_loc[0] - size) + abs(king_loc[1] - 0)
    else:
        raise Exception("Unexpectedly failed to calculate King's distance to corner.")


def get_mobility(legal_moves: int,
                 player: str) -> int:
    """
    Return the mobility of the player. This is player's starting moves - player's current moves.

    This doesn't calculate legal moves since that is expensive and is already done each turn anyway. It should
    be cached and passed to this function instead of calculated twice.

    :param legal_moves: The number of legal moves the player has.
    :param player: The player whose mobility we're checking. Either "defenders" or "attackers".
    :return: The mobility value.
    """

    # This assumes the number of moves is for 11x11 Copenhagen Hnefatafl. It does not generalize.
    if player == 'attackers':
        return 116 - legal_moves
    else:
        return 60 - legal_moves


def get_escorts(board: np.array) -> int:
    """
    Returns the number of escorts the King has.

    The King has four adjacent sides. An escort is defined as follows: if the King can move in a straight line and
    bump into a defender, that direction is covered by an escort. The reasoning behind this is that if the King can
    bump into a defender, then that logically means that any attacked who moves adjacent to the King on that side
    could be immediately captured by moving the escorting defender and flanking the attacker between the defender
    and the King. Therefore, this is a simple measure of King safety.

    Note that to avoid penalizing the defenders for having the King on an edge, the edge's side is considered to be
    escorted if the King is directly on the edge (because an attacker cannot attack from that direction).

    :param board: The 3D NumPy "board" array on which the game is being played.
    :return: The number of escorted sides. An integer from 0 to 4.
    """
    # Get the King's location and board size
    king_loc = find_king(board)
    size = board.shape[2] - 1

    # Initialize number of escorts
    escorts = 0

    # For every direction, check for an escort.
    for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        row, col = king_loc
        row += direction[0]
        col += direction[1]
        if not in_bounds(row, col, size):
            escorts += 1
            continue

        while True:
            if not in_bounds(row, col, size):
                break
            # NOTE, technically if the defender is exactly 2 squares from the King, the King is not safe in that
            # direction since an attacker could move between them without penalty.
            elif is_defender(board, row, col):
                escorts += 1
                break
            elif is_attacker(board, row, col):
                break
            row += direction[0]
            col += direction[1]
    return escorts


def get_attack_options(board: np.array,
                       ) -> int:
    """
    Return the number of attackers who could move adjacent to the King at any given time. This loosely encodes
    the number of options the attackers have in how they try to capture the King.

    :param np.array board: The 3D NumPy "board" array on which the game is played.
    :return: The number of attackers who could move adjacent to the King.
    """
    # Get the King's location and board size
    king_loc = find_king(board)
    size = board.shape[2] - 1

    # Initialize number of attack options
    attacks = 0

    # The list of blank tiles next to the King that could be "attacked".
    vulnerable_tiles = []

    # Get the tiles adjacent to the King that are blank
    for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        row, col = king_loc
        row += direction[0]
        col += direction[1]
        if is_blank_thin(board, row, col, size):
            vulnerable_tiles.append((row, col))

    for tile in vulnerable_tiles:
        # For every direction, check for an attacker.
        for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            row, col = tile
            while True:
                row += direction[0]
                col += direction[1]
                if not in_bounds(row, col, size):
                    break
                elif is_attacker(board, row, col):
                    attacks += 1
                    break
                elif is_piece_thin(board, row, col):
                    break
    return attacks


def get_map_control(board: np.array,
                    ) -> int:
    """
    Returns the net "map control".

    Map control is defined as the number of rows and columns controlled by a player. A player is defined to have
    "control" over a row/column if the same player has the piece at the lowest and highest index of the row/column.
    More intuitively, if the last pieces on the "ends" of any row and column belong to the same player, that player
    controls the row/column.

    The net map control is number of rows/columns controlled by defenders - the row/columns controlled by attackers.

    The idea is that attackers want to have general, broad map control in order to encircle the defenders, and the
    defenders want to have some map control to avoid this.

    :param np.array board: The 3D NumPy "board" array on which the game is played.
    :return: An integer representing the net map control.
    """

    size = board.shape[1]

    # Initialize control to 0
    defender_controlled = 0
    attacker_controlled = 0

    for row in range(size):
        first = None
        last = None
        # Iterate low to high to find the first piece (if any)
        for i in range(0, size, 1):
            if is_attacker(board, row, i):
                first = 'attacker'
                break
            elif is_defender(board, row, i):
                first = 'defender'
                break

        for i in range(size - 1, 0, -1):
            if is_attacker(board, row, i):
                last = 'attacker'
                break
            elif is_defender(board, row, i):
                last = 'defender'
                break
        if first and last and first is last:
            if first == 'attacker':
                attacker_controlled += 1
            else:
                defender_controlled += 1

    for col in range(size):
        first = None
        last = None
        # Iterate low to high to find the first piece (if any)
        for i in range(0, size, 1):
            if is_attacker(board, i, col):
                first = 'attacker'
                break
            elif is_defender(board, i, col):
                first = 'defender'
                break

        # Iterate high to low to find the last piece (if any)
        for i in range(size - 1, 0, -1):
            if is_attacker(board, i, col):
                last = 'attacker'
                break
            elif is_defender(board, i, col):
                last = 'defender'
                break
        if first and last and first == last:
            if first == 'attacker':
                attacker_controlled += 1
            else:
                defender_controlled += 1
    net_control = defender_controlled - attacker_controlled
    return net_control


def get_close_pieces(board: np.array,
                     pieces: str = 'defenders',
                     ) -> int:
    """
    Count the defenders adjacent to the King.

    :param board:
    :param pieces: Which pieces around the king are we looking for?
    :return: The number of defenders adjacent to the King.
    """

    # Get the King's location and board size
    king_loc = find_king(board)
    size = board.shape[2] - 1

    # Initialize number of escorts
    close_pieces = 0

    # For every direction, check for an escort.
    for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        row, col = king_loc
        row += direction[0]
        col += direction[1]
        if pieces == 'defenders':
            if in_bounds(row, col, size) and is_defender(board, row, col):
                close_pieces += 1
        else:
            if in_bounds(row, col, size) and is_attacker(board, row, col):
                close_pieces += 1
    return close_pieces


def king_can_escape(board: np.array,
                    piece_flags: np.array,
                    ) -> int:
    """

    :param board:
    :param piece_flags:
    :return:
    """

    king_loc = find_king(board)
    size = board.shape[2] - 1
    if not is_edge(king_loc[0], king_loc[1], size):
        return 0

    elif king_loc[0] == 0 or king_loc[0] == size:
        # King is on top/bottom edge, check if he can walk to left or right corner without hitting a piece
        row, col = king_loc
        while col > 0:
            col -= 1
            if not is_blank(row, col, size, piece_flags):
                break
            elif col == 0:
                return 1
        row, col = king_loc
        while col < size:
            col += 1
            if not is_blank(row, col, size, piece_flags):
                break
            elif col == size:
                return 1

    elif king_loc[1] == 0 or king_loc[1] == size:
        # King is on left/right edge, check if he can walk to left or right corner without hitting a piece
        row, col = king_loc
        while row > 0:
            row -= 1
            if not is_blank(row, col, size, piece_flags):
                break
            elif row == 0:
                return 1
        row, col = king_loc
        while row < size:
            row += 1
            if not is_blank(row, col, size, piece_flags):
                break
            elif row == size:
                return 1
    return 0


def extract_features(board: np.array,
                     defender_moves: int,
                     attacker_moves: int,
                     piece_flags: np.array,
                     ) -> dict:
    """
    :param board:
    :param defender_moves:
    :param attacker_moves:
    :param piece_flags:
    :return:
    """
    size = board.shape[1] - 1
    material_balance = get_material_balance(board)
    king_dist = get_king_distance_to_corner(board)
    escorts = get_escorts(board)
    attack_options = get_attack_options(board)
    close_defenders = get_close_pieces(board, 'defenders')
    close_attackers = get_close_pieces(board, 'attackers')
    attacker_mobility = get_mobility(attacker_moves, 'attackers')
    defender_mobility = get_mobility(defender_moves, 'defenders')
    mobility_delta = defender_mobility - attacker_mobility
    map_control = get_map_control(board)
    king_escape = king_can_escape(board, piece_flags)
    king_loc = find_king(board)
    king_edge = 1 if is_edge(king_loc[0], king_loc[1], size) else 0
    payload = {'material_balance': material_balance,
               'king_dist': king_dist,
               'escorts': escorts,
               'attack_options': attack_options,
               'close_defenders': close_defenders,
               'close_attackers': close_attackers,
               'mobility_delta': mobility_delta,
               'map_control': map_control,
               'king_escape': king_escape,
               'king_edge': king_edge
               }
    return payload
# Feature engineering functions end here


def get_moves(board: np.array,
              index: Tuple[int, int],
              cache: np.array,
              dirty_map: dict,
              dirty_flags: set,
              piece_flags: np.array
              ) -> np.array:
    """
    Return a binary array of legal moves.

    The index of the array encodes a move relative to the passed in index.
    The value of the array indicates whether that move is legal (1) or not (0).
    Indices 0-9 encode "move up 1-10 spaces", indices 10-19 encode "move down 1-10 spaces",
    Indices 20-29 encode "move left 1-10 spaces", indices 30-39 encode "move right 1-10 spaces".

    :param np.array board: a 2D NumPy array representing the board.
    :param tuple index: a tuple(int, int) representing the index of the piece whose legal moves we're checking.
    :param np.array cache: a reference to an array of the cached action space.
    :param dict dirty_map: a dict mapping index i to a list of indices whose caches are invalidate if i moves.
    :param np.array dirty_flags: a 2D array of flags indicating whether the move cache of (row, col) is dirty.
    :param np.array piece_flags: A 2D binary NumPy array. If (row, col) is 1, a piece if present, otherwise no piece.
    :return: A 1D binary NumPy array of length 40 representing legal moves.
    """

    # If there is no piece on this tile, return immediately (there are no legal moves if there's no piece!)
    if not is_piece(index[0], index[1], piece_flags):
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
            if not is_piece(tmp_index[0], tmp_index[1], piece_flags):
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
                        piece_flags: np.array,
                        ) -> None:
    """
    Refresh any dirty cache locations.

    :param np.array board: The 3D NumPy array "board" on which the game is being played.
    :param np.array cache: The 3D NumPy array cache of moves.
    :param dirty_map: A dictionary mapping index value i to a list of indices j that would experience cache invalidation
                      if i moves, e.g. if i moves, the legal moves for every j need to be refreshed.
    :param dirty_flags: A set of tuples that need to have their legal move cache refreshed.
    :param np.array piece_flags: A 2D binary NumPy array. If (row, col) is 1, a piece if present, otherwise no piece.
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
                      piece_flags,
                      )


def has_moves(board: np.array,
              cache: np.array,
              dirty_map: dict,
              dirty_flags: set,
              player: str,
              piece_flags: np.array,
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
    :param np.array piece_flags: A 2D binary NumPy array. If (row, col) is 1, a piece if present, otherwise no piece.
    :return: True if player has remaining moves, False if player has no legal moves.
    """
    update_action_space(board=board, cache=cache, dirty_map=dirty_map, dirty_flags=dirty_flags, piece_flags=piece_flags)
    if player == 'attackers':
        mask = board[0, :, :] == 1
    else:
        mask = np.sum(board[1:, :, :], axis=0) == 1
    return cache[:, mask].any()


def all_legal_moves(board: np.array,
                    cache: np.array,
                    dirty_map: dict,
                    dirty_flags: set,
                    player: str,
                    piece_flags: np.array,
                    ) -> np.array:
    """
    Return a 3D NumPy array representing the action-space of a single player.

    :param np.array board: The 3D NumPy array "board" on which the game is being played.
    :param np.array cache: The 3D NumPy array cache of moves.
    :param dirty_map: A dictionary mapping index value i to a list of indices j that would experience cache invalidation
                      if i moves, e.g. if i moves, the legal moves for every j need to be refreshed.
    :param dirty_flags: A set of tuples that need to have their legal move cache refreshed.
    :param str player: The player. It is either "attackers" or "defenders".
    :param np.array piece_flags: A 2D binary NumPy array. If (row, col) is 1, a piece if present, otherwise no piece.
    :return: A 3D NumPy array representing the action-space of a single player. For standard hnefatafl, it is 40x11x11.
    """
    update_action_space(board=board, cache=cache, dirty_map=dirty_map, dirty_flags=dirty_flags, piece_flags=piece_flags)
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
              piece_flags: np.array,
              thin_move: bool = False,
              ) -> tuple:
    """
    Move the piece at index according to move.

    :param np.array board: The 3D NumPy array "board" on which the game is being played.
    :param Tuple[int, int] index: The index of the piece to be moved.
    :param int move: The encoded move that the piece at index will make.
    :param np.array cache: The 3D NumPy array cache of moves.
    :param dirty_map: A dictionary mapping index value i to a list of indices j that would experience cache invalidation
                      if i moves, e.g. if i moves, the legal moves for every j need to be refreshed.
    :param dirty_flags: A set of tuples that need to have their legal move cache refreshed.
    :param np.array piece_flags: A 2D binary NumPy array. If (row, col) is 1, a piece if present, otherwise no piece.
    :param bool thin_move: A boolean. If true, caches won't be updated (except piece_flags). This is used for action evaluation.
    :return: The new (row, col) tuple that the piece moved to from the old index.
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

    # Ensure there isn't a piece already at the destination
    if is_piece(new_index[0], new_index[1], piece_flags):
        raise Exception(f"Tried move from {index[0]}, {index[1]} to {new_index[0]}, {new_index[1]}, but it's occupied.")

    board[plane, new_index[0], new_index[1]] = 1
    board[plane, index[0], index[1]] = 0

    # Update piece_flags. This is a reference to the cache of piece locations.
    piece_flags[index[0], index[1]] = 0
    piece_flags[new_index[0], new_index[1]] = 1

    if not thin_move:

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
                      piece_flags
                      )

        # Update the cache of the indices affected by the move to the new location
        for affected_index in dirty_map[new_index]:
            dirty_flags.add(affected_index)

    if thin_move:
        return new_index, index

    return new_index


def revert_move(board: np.array,
                new_index: Tuple[int, int],
                old_index: Tuple[int, int],
                piece_flags: np.array,
                ) -> None:
    """
    Reverts a thin_move. This is used during move evaluation.

    :param board:
    :param new_index:
    :param old_index:
    :param piece_flags:
    :return:
    """

    if board[0, new_index[0], new_index[1]] == 1:
        plane = 0
    elif board[1, new_index[0], new_index[1]] == 1:
        plane = 1
    else:
        plane = 2

    board[plane, old_index[0], old_index[1]] = 1
    board[plane, new_index[0], new_index[1]] = 0

    # Update piece_flags. This is a reference to the cache of piece locations.
    piece_flags[old_index[0], old_index[1]] = 1
    piece_flags[new_index[0], new_index[1]] = 0


def check_capture(board: np.array,
                  index: tuple,
                  piece_flags: np.array,
                  dirty_map: dict,
                  dirty_flags: set,
                  cache: np.array,
                  thin_capture: bool = False,
                  ) -> int:
    """
    Given an index, checks to see if any basic enemies pieces around it are captured.

    :param np.array board: The 3D NumPy array "board" on which the game is being played.
    :param Tuple[int, int] index: The index of the piece around which we check for pieces to capture.
    :param dict dirty_map:
    :param set dirty_flags:
    :param np.array cache:
    :param np.array piece_flags: A 2D binary NumPy array. If (row, col) is 1, a piece if present, otherwise no piece.
    :param bool thin_capture: A boolean. If true, captures will be counted, but not actually taken.
    :return: Integer number of pieces captured.
    """
    # Set up some convenient variables
    row, col, teams, size, hostile, plane, ally = get_nice_variables(board, index)

    # If the throne is empty, it is hostile
    if not board[2, size // 2, size // 2] == 2:
        hostile.add((size // 2, size // 2))

    captures = 0

    # All of these if statements could probably be collapsed in a similar way as check_shield_wall()
    if row > 0 and is_enemy(board, row - 1, col, ally, piece_flags):
        if is_edge(row - 1, col, size):
            tags = []
            if check_shield_wall(board, (row - 1, col), tags, piece_flags, ally):
                captures += len(tags)
                if not thin_capture:
                    capture_tags(board, tags, piece_flags=piece_flags,
                                 cache=cache, dirty_flags=dirty_flags, dirty_map=dirty_map)
        # if the enemy is not on an edge, and the other side is an allied piece or hostile piece
        if row - 2 >= 0 and is_flanked(board, row - 2, col, ally, hostile, piece_flags):
            # Destroy it!
            captures += 1
            if not thin_capture:
                board[:, row - 1, col] = 0
                piece_flags[row - 1, col] = 0
                # TAG NEW DIRTY FLAGS HERE!
                for affected_index in dirty_map[(row - 1, col)]:
                    dirty_flags.add(affected_index)
                # UPDATE THE CACHE AT THE CAPTURE LOCATION TO REMOVE LEGAL ACTIONS
                _ = get_moves(board,
                              (row - 1, col),
                              cache,
                              dirty_map,
                              dirty_flags,
                              piece_flags
                              )

    if row < size and is_enemy(board, row + 1, col, ally, piece_flags):
        if is_edge(row + 1, col, size):
            tags = []
            if check_shield_wall(board, (row + 1, col), tags, piece_flags, ally):
                captures += len(tags)
                if not thin_capture:
                    capture_tags(board, tags, piece_flags=piece_flags,
                                 cache=cache, dirty_flags=dirty_flags, dirty_map=dirty_map)
        if row + 2 <= size and is_flanked(board, row + 2, col, ally, hostile, piece_flags):
            captures += 1
            if not thin_capture:
                board[:, row + 1, col] = 0
                piece_flags[row + 1, col] = 0
                # TAG NEW DIRTY FLAGS HERE!
                for affected_index in dirty_map[(row + 1, col)]:
                    dirty_flags.add(affected_index)
                # UPDATE THE CACHE AT THE CAPTURE LOCATION TO REMOVE LEGAL ACTIONS
                _ = get_moves(board,
                              (row + 1, col),
                              cache,
                              dirty_map,
                              dirty_flags,
                              piece_flags
                              )

    if col > 0 and is_enemy(board, row, col - 1, ally, piece_flags):
        if is_edge(row, col - 1, size):
            tags = []
            if check_shield_wall(board, (row, col - 1), tags, piece_flags, ally):
                captures += len(tags)
                if not thin_capture:
                    capture_tags(board, tags, piece_flags=piece_flags,
                                 cache=cache, dirty_flags=dirty_flags, dirty_map=dirty_map)
        if col - 2 >= 0 and is_flanked(board, row, col - 2, ally, hostile, piece_flags):
            captures += 1
            if not thin_capture:
                board[:, row, col - 1] = 0
                piece_flags[row, col - 1] = 0
                # TAG NEW DIRTY FLAGS HERE!
                for affected_index in dirty_map[(row, col - 1)]:
                    dirty_flags.add(affected_index)
                # UPDATE THE CACHE AT THE CAPTURE LOCATION TO REMOVE LEGAL ACTIONS
                _ = get_moves(board,
                              (row, col - 1),
                              cache,
                              dirty_map,
                              dirty_flags,
                              piece_flags
                              )

    if col < size and is_enemy(board, row, col + 1, ally, piece_flags):
        if is_edge(row, col + 1, size):
            tags = []
            if check_shield_wall(board, (row, col + 1), tags, piece_flags, ally):
                captures += len(tags)
                if not thin_capture:
                    capture_tags(board, tags, piece_flags=piece_flags,
                                 cache=cache, dirty_flags=dirty_flags, dirty_map=dirty_map)
        if col + 2 <= size and is_flanked(board, row, col + 2, ally, hostile, piece_flags):
            captures += 1
            if not thin_capture:
                board[:, row, col + 1] = 0
                piece_flags[row, col + 1] = 0
                # TAG NEW DIRTY FLAGS HERE!
                for affected_index in dirty_map[(row, col + 1)]:
                    dirty_flags.add(affected_index)
                # UPDATE THE CACHE AT THE CAPTURE LOCATION TO REMOVE LEGAL ACTIONS
                _ = get_moves(board,
                              (row, col + 1),
                              cache,
                              dirty_map,
                              dirty_flags,
                              piece_flags
                              )

    return captures


def capture_tags(board: np.array,
                 tags: list,
                 piece_flags: np.array,
                 cache: np.array,
                 dirty_flags: set,
                 dirty_map: dict,
                 ) -> None:
    """
    Capture any non-King pieces who are "tagged" as being trapped in a shield wall.

    :param np.array board: The 3D NumPy array "board" on which the game is being played.
    :param list tags: A list of tuples. Each tuple is a piece that was tagged as being trapped in a shield wall.
    :param np.array piece_flags: A 2D binary NumPy array. If (row, col) is 1, a piece if present, otherwise no piece.
    :param cache:
    :param dirty_flags:
    :param dirty_map:
    :return: None; any enemies who are trapped in a shield wall are eliminated by this function.
    """
    for tag in tags:
        if np.argwhere(board[:, tag[0], tag[1]] == 1).item() != 2:
            board[:, tag[0], tag[1]] = 0
            piece_flags[tag[0], tag[1]] = 0
            # TAG NEW DIRTY FLAGS HERE!
            for affected_index in dirty_map[tag]:
                dirty_flags.add(affected_index)
            # UPDATE THE CACHE AT THE CAPTURE LOCATION TO REMOVE LEGAL ACTIONS
            _ = get_moves(board,
                          tag,
                          cache,
                          dirty_map,
                          dirty_flags,
                          piece_flags
                          )


def check_shield_wall(board: np.array,
                      index: Tuple[int, int],
                      tags: list,
                      piece_flags: np.array,
                      ally: int,
                      ) -> bool:
    """
    Check whether a shield wall has been created

    :param np.array board: The 3D NumPy 'board' array on which the game is being played.
    :param index:
    :param tags:
    :param np.array piece_flags: A 2D binary NumPy array. If (row, col) is 1, a piece if present, otherwise no piece.
    :param ally:
    :return: True if a shield wall capture has occurred.
    """
    size = board.shape[1] - 1
    queue = deque()
    visited = set()

    # The initial tile in the queue is the enemy being checked first
    queue.append(index)
    tags.append(index)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        row, col = queue.popleft()
        visited.add((row, col))
        if not is_edge(row, col, size):
            return False
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if is_blank(nr, nc, size, piece_flags) and not is_corner(nr, nc, size):
                return False
            elif (in_bounds(nr, nc, size) and
                  not is_ally(board, nr, nc, ally, piece_flags) and
                  not is_corner(nr, nc, size) and
                  (nr, nc) not in visited
            ):
                queue.append((nr, nc))
                tags.append((nr, nc))
    return True


def is_fort(board: np.array,
            index: Tuple[int, int],
            defender_tags: List[Tuple[int, int],],
            interior_tags: List[Tuple[int, int],],
            piece_flags: np.array,
            ) -> bool:
    """
    Check whether the King is in an edge fort.

    :param np.array board: The 3D NumPy "board" array on which the game is being played.
    :param Tuple[int, int] index: The current index of the tile being checked with fort logic.
    :param list defender_tags: A list of defender indices. If this is a fort, then these are the "walls" of the fort.
    :param list interior_tags: A list of interior tiles. If this is a fort, then these are "inside" the fort.
    :param np.array piece_flags: A 2D binary NumPy array. If (row, col) is 1, a piece if present, otherwise no piece.
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
        elif is_blank(row + step[0], col + step[1], size, piece_flags) or is_king(board, row + step[0], col + step[1]):
            if (row + step[0], col + step[1]) not in interior_tags:
                adjacent_interior.append((row + step[0], col + step[1]))
        elif is_defender(board, row + step[0], col + step[1]):
            if (row + step[0], col + step[1]) not in defender_tags:
                defender_tags.append((row + step[0], col + step[1]))
        else:
            return False

    for tile in adjacent_interior:
        if not is_fort(board, tile, defender_tags, interior_tags, piece_flags):
            return False
    return True


def verify_encirclement(board: np.array,
                        piece_flags: np.array,
                        ) -> Tuple[list, list]:
    """
    Return the "walls" of an encirclement.

    The function check_encirclement() doesn't record the attackers who form the encirclement, because almost always
    there won't be an encirclement. So it chooses to quickly try to dismiss encirclement and not contrive to figure out
    attackers who may form an encirclement. If it turns out that there is an encirclement, this algorithm efficiently
    uses a queue-based flood fill from the defenders outward to find the attackers who make up the "wall" around them.
    Then we use is_impenetrable with option='encirclement' to confirm that it is a game-ending encirclement that
    can't be escaped.

    :param np.array board: The 3D NumPy "board" array on which the game is being played.
    :param np.array piece_flags: A 2D binary NumPy array. If (row, col) is 1, a piece if present, otherwise no piece.
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
                if is_defender(board, nr, nc) or is_blank(nr, nc, size, piece_flags):
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


def check_encirclement(board: np.array,
                       piece_flags: np.array,
                       ) -> bool:
    """
    Check whether the attackers have encircled all defenders.

    :param np.array board: The 3D NumPy 'board' array on which the game is being played.
    :param np.array piece_flags: A 2D binary NumPy array. If (row, col) is 1, a piece if present, otherwise no piece.
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
                elif is_blank(nr, nc, size, piece_flags):
                    queue.append((nr, nc))

    return True


def check_king(board: np.array,
               piece_flags: np.array,
               ) -> int:
    """
    Check whether the King has escaped or been captured.

    :param np.array board: A 2D NumPy array representing the board.
    :param np.array piece_flags: A 2D binary NumPy array. If (row, col) is 1, a piece if present, otherwise no piece.
    :return: -1 means King captured, 1 means King escaped, 0 means neither.
    """
    size = board.shape[1] - 1
    row, col = find_king(board)
    corners = [(0, 0), (0, size), (size, 0), (size, size)]
    throne = [(size // 2, size // 2)]
    hostile = corners + throne

    # Has the King escaped?
    if (row, col) in corners:
        return 1

    # Is the king surrounded?
    if ((in_bounds(row - 1, col, size) and (is_attacker(board, row - 1, col) or (row - 1, col) in hostile)) and
        (in_bounds(row + 1, col, size) and (is_attacker(board, row + 1, col) or (row + 1, col) in hostile)) and
        (in_bounds(row, col - 1, size) and (is_attacker(board, row, col - 1) or (row, col - 1) in hostile)) and
        (in_bounds(row, col + 1, size) and (is_attacker(board, row, col + 1) or (row, col + 1) in hostile))
       ):
        return -1
    return 0


def is_terminal(board: np.array,
                cache: np.array,
                dirty_map: dict,
                dirty_flags: set,
                player: str,
                piece_flags: np.array,
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
    :param np.array piece_flags: A 2D binary NumPy array. If (row, col) is 1, a piece if present, otherwise no piece.
    :param int attacker_moves: The number of legal moves that the attackers had as of the latest check.
    :param int defender_moves: The number of legal moves that the attackers had as of the latest check.
    :return: "defenders" or "attackers" if either has won, otherwise None.
    """
    king_state = check_king(board, piece_flags)
    if king_state == 1:
        print("King escape detected.")
        return "defenders", "king_escaped"
    elif king_state == -1:
        print("King capture detected.")
        return "attackers", "king_captured"
    elif player == "defenders":
        king_r, king_c = find_king(board)
        defender_tags = []
        interior_tags = []
        if (is_edge(king_r, king_c, board.shape[1] - 1) and
           is_fort(board, (king_r, king_c), defender_tags, interior_tags, piece_flags) and
           is_impenetrable(board, defender_tags, interior_tags)):
            print("Exit Fort detected.")
            return "defenders", "exit_fort"
    elif player == "attackers":
        if check_encirclement(board, piece_flags):
            attacker_walls, visited = verify_encirclement(board, piece_flags)
            if is_impenetrable(board, attacker_walls, visited, option='encirclement'):
                print("Encirclement detected.")
                return "attackers", "encirclement"
    # Players already check their legal moves each turn, so this is a redundant (and expensive) operation.
    # Optimize this by just passing in the known number of legal moves. If it's zero, that player loses.
    if defender_moves < 10 and not has_moves(board=board, cache=cache, dirty_map=dirty_map,
                                             dirty_flags=dirty_flags, player="defenders", piece_flags=piece_flags):
        #print("The defenders have no legal moves.")
        return "attackers", "defenders_no_moves"
    elif attacker_moves < 10 and not has_moves(board=board, cache=cache, dirty_map=dirty_map,
                                               dirty_flags=dirty_flags, player="attackers", piece_flags=piece_flags):
        #print("The attackers have no legal moves.")
        return "defenders", "attackers_no_moves"

    # Does not check for draws
    else:
        return "n/a", "n/a"
