import game_logic
import hnefatafl
import random
import numpy as np

CHAR_MAP = {0: '.',
            1: 'A',
            2: 'D',
            3: 'K',
            }


def np_to_string(board):
    board_string = ""
    for i in range(11):
        if i != 0:
            board_string += '\n'
        for j in range(11):
            board_string += CHAR_MAP[board[i, j]]
    return board_string


def fill_board(board, k1, k2):
    for i in range(k1):
        x, y = random.randint(0, 10), random.randint(0, 10)
        while board[x, y] != 0:
            x, y = random.randint(0, 10), random.randint(0, 10)
        board[x, y] = 1

    for i in range(k2):
        x, y = random.randint(0, 10), random.randint(0, 10)
        while board[x, y] != 0:
            x, y = random.randint(0, 10), random.randint(0, 10)
        board[x, y] = 2


def generate_defender_curriculum(n, k1, k2):
    curriculum_board = np.zeros((11, 11), dtype=int)
    cornerx, cornery = random.choice([0, 10]), random.choice([0, 10])
    curriculum_board[cornerx, cornery] = 3

    fill_board(curriculum_board, k1, k2)
    board_string = np_to_string(curriculum_board)
    curriculum_game = hnefatafl.initialize_game(board_string)

    for i in range(n):
        legal_moves = game_logic.all_legal_moves(board=curriculum_game.board,
                                                 cache=curriculum_game.cache,
                                                 dirty_map=curriculum_game.dirty_map,
                                                 dirty_flags=curriculum_game.dirty_flags,
                                                 player='defenders',
                                                 piece_flags=curriculum_game.piece_flags)

        if i == 0:
            try:
                legal_indices = np.where(legal_moves[:, cornerx, cornery] == 1)
                choices = len(legal_indices[0])
                choice = np.random.choice(choices)
                move, row, col = legal_indices[0][choice], cornerx, cornery
            except:
                legal_indices = np.where(legal_moves == 1)
                choices = len(legal_indices[0])
                choice = np.random.choice(choices)
                move, row, col = legal_indices[0][choice], legal_indices[1][choice], legal_indices[2][choice]
        else:
            legal_indices = np.where(legal_moves == 1)
            choices = len(legal_indices[0])
            choice = np.random.choice(choices)
            move, row, col = legal_indices[0][choice], legal_indices[1][choice], legal_indices[2][choice]

        new_index = game_logic.make_move(curriculum_game.board,
                                         (row, col),
                                         move,
                                         cache=curriculum_game.cache,
                                         dirty_map=curriculum_game.dirty_map,
                                         dirty_flags=curriculum_game.dirty_flags,
                                         piece_flags=curriculum_game.piece_flags,)
    return curriculum_game


def generate_attacker_curriculum(n, k1, k2):
    curriculum_board = np.zeros((11, 11), dtype=int)

    # Generate the king's location. Cannot be on an edge or corner since he's going to be surrounded.
    kingx, kingy = 5, 5#random.randint(1, 9), random.randint(1, 9)
    curriculum_board[kingx, kingy] = 3

    # Surround the king
    curriculum_board[kingx - 1, kingy] = 1
    curriculum_board[kingx + 1, kingy] = 1
    curriculum_board[kingx, kingy - 1] = 1
    curriculum_board[kingx, kingy + 1] = 1

    # Randomly select the index of one of the surrounded attackers since he's going to back off first
    attackerx, attackery = (kingx, kingy - 1)#random.choice([(kingx, kingy - 1),
    #                                       (kingx, kingy + 1),
    #                                       (kingx + 1, kingy),
    #                                       (kingx + 1, kingy)])

    fill_board(curriculum_board, k1, k2)
    board_string = np_to_string(curriculum_board)
    curriculum_game = hnefatafl.initialize_game(board_string)

    for i in range(n):
        legal_moves = game_logic.all_legal_moves(board=curriculum_game.board,
                                                 cache=curriculum_game.cache,
                                                 dirty_map=curriculum_game.dirty_map,
                                                 dirty_flags=curriculum_game.dirty_flags,
                                                 player='attackers',
                                                 piece_flags=curriculum_game.piece_flags)

        if i == 0:
            try:
                legal_indices = np.where(legal_moves[:, attackerx, attackery] == 1)
                choices = len(legal_indices[0])
                choice = np.random.choice(choices)
                move, row, col = legal_indices[0][choice], attackerx, attackery
            except:
                legal_indices = np.where(legal_moves == 1)
                choices = len(legal_indices[0])
                choice = np.random.choice(choices)
                move, row, col = legal_indices[0][choice], legal_indices[1][choice], legal_indices[2][choice]
        else:
            legal_indices = np.where(legal_moves == 1)
            choices = len(legal_indices[0])
            choice = np.random.choice(choices)
            move, row, col = legal_indices[0][choice], legal_indices[1][choice], legal_indices[2][choice]

        new_index = game_logic.make_move(curriculum_game.board,
                                         (row, col),
                                         move,
                                         cache=curriculum_game.cache,
                                         dirty_map=curriculum_game.dirty_map,
                                         dirty_flags=curriculum_game.dirty_flags,
                                         piece_flags=curriculum_game.piece_flags,)
    return curriculum_game

