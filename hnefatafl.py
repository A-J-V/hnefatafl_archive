from copy import deepcopy
import numpy as np
from utilities import *
from itertools import product

# Store the initial map in a string
starting_board = """\
...AAAAA...
.....A.....
...........
A....D....A
A...DDD...A
AA.DDKDD.AA
A...DDD...A
A....D....A
...........
.....A.....
...AAAAA..."""


class TaflBoard:
    def __init__(self, starting_board: str = starting_board) -> None:
        """
        Create the game board.

        Takes a string representation of a Tafl board and encodes it into a 3D NumPy array.
        The board being received must have length == width, and length must be an odd number.

        :param str starting_board:
        """
        assert len(starting_board.replace('\n',
                                          '')) ** 0.5 % 1 == 0, """The provided starting board
                                           is not square. A Hnefatafl board must be square!"""
        # Create three binary spatial planes, one plane for each piece type
        a_plane = np.array([[1 if i == 'A' else 0 for i in list(c)] for c in starting_board.splitlines()])
        d_plane = np.array([[1 if i == 'D' else 0 for i in list(c)] for c in starting_board.splitlines()])
        k_plane = np.array([[1 if i == 'K' else 0 for i in list(c)] for c in starting_board.splitlines()])
        # Stack the planes into an array, M x N x N, where N is board length/width and M is number of piece types
        self.board = np.stack([a_plane, d_plane, k_plane])
        self.shape = self.board.shape[-1]
        # Define a set of dirty flags
        self.dirty_flags = set(product(list(range(self.shape)), list(range(self.shape))))
        # Define a dictionary mapping index i to a list of indices whose caches are invalidated when i moves
        self.dirty_map = {(row, col): [] for (row, col) in product(list(range(self.shape)), list(range(self.shape)))}
        self.cache = np.zeros(shape=(40, self.shape, self.shape))
        self.piece_flags = np.sum(self.board, axis=0)
        set_global_piece_flags(piece_flags=self.piece_flags)
        self.game_over = False

    def __repr__(self, board_array: np.array = None) -> str:
        """Return a more human-readable string of the game board."""
        if board_array is None:
            board_array = self.board
        tmp = deepcopy(board_array)
        for i in range(3):
            tmp[i, :, :] *= (i + 1)
        tmp = tmp.sum(axis=0).astype('str')
        for i, piece in enumerate(['.', 'A', 'D', 'K', '+']):
            tmp[tmp == str(i)] = piece
        tmp = np.pad(tmp, pad_width=1, constant_values='X')
        tmp = str(tmp)
        tmp = tmp.replace("\'", '').replace('[', ' ').replace(']', '')
        return tmp

    def display_moves(self, index: tuple) -> None:
        """
        Print an ASCII representation of the map  with '+' showing valid moves.

        :param tuple index: A tuple (row, column) of the piece whose valid moves we're checking.
        """
        moves = get_moves(self.board, index)
        tmp = deepcopy(self.board)
        for k, instruction in enumerate([(0, -1), (0, 1), (1, -1), (1, 1)]):
            axis, direction = instruction
            tmp_index = list(index)
            i = k * 10
            while i < (k + 1) * 10:
                tmp_index[axis] += direction
                if moves[i] == 1:
                    tmp[0, tmp_index[0], tmp_index[1]] = 4
                i += 1
        tmp = self.__repr__(tmp)
        print(tmp)
