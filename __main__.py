import graphics
import hnefatafl
import pygame
import time
from utilities import *
from MCTS import simulate
import cProfile
import random


from testing import run_tests
from test_boards import test_boards as tb


if __name__ == "__main__":
    b = hnefatafl.TaflBoard()
    #actions = all_legal_moves(board=b.board, cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags, player="attackers")
    #actions = np.argwhere(actions == 1)
    #actions = [(move, row, col) for move, row, col in actions]
    #random.shuffle(actions)
    #print([(move, row, col) for move, row, col in actions])
    #simulate(b.board,
    #         b.cache,
    #         b.dirty_map,
    #         b.dirty_flags,
    #         'attackers',
    #         b.piece_flags,
    #         visualize=True)
    #print(b.dirty_flags)
    #cProfile.run("simulate(b.board, b.cache, b.dirty_map, b.dirty_flags, 'attackers', piece_flags=b.piece_flags, visualize=False)")
    #times = []
    #turns = []
    #wins = []
    #for _ in range(10000):
    #    b = hnefatafl.TaflBoard()
    #    start = time.time()
    #    t, v = simulate(b.board,
    #                    b.cache,
    #                    b.dirty_map,
    #                    b.dirty_flags,
    #                    'attackers',
    #                    b.piece_flags,
    #                    visualize=False)
    #    times.append(time.time() - start)
    #    turns.append(t)
    #    wins.append(v)
    #print(f"Average rollout time: {sum(times) / len(times)}")
    #print(f"Average rollout turns: {sum(turns) / len(turns)}")
    #print(f"Defender winrate: {sum(wins) / len(wins)}")

    #display = graphics.initialize()
    #graphics.refresh(b.board, display)
    #time.sleep(5)
    #while True:
    #    piece = input("Choose a piece: row, col: ")
    #    piece = piece.split(' ')
    #    row, col = int(piece[0]), int(piece[1])
    #    target = input("Choose a target: move int: ")
    #    target = int(target)
    #    make_move(b.board, (row, col), target)
    #    graphics.refresh(b.board, display)

    # Need to add many tests that check not just a static state, but the transition between states.
    # These tests have a starting board, then make a move, and check that the state after the move
    # such as the board and legal moves and dirty flags are all as expected.
    run_tests()
