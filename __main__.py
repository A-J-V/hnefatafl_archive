import graphics
import hnefatafl
import pygame
import time
from utilities import *
from MCTS import simulate
import cProfile


from testing import run_tests
from test_boards import test_boards as tb


if __name__ == "__main__":
    b = hnefatafl.TaflBoard()
    #simulate(b.board,
    #         b.cache,
    #         b.dirty_map,
    #         b.dirty_flags,
    #         'attackers',
    #         visualize=True)
    #print(b.dirty_flags)
    #cProfile.run("simulate(b.board, b.cache, b.dirty_map, b.dirty_flags, 'attackers', visualize=False)")
    times = []
    turns = []
    for _ in range(1000):
        b = hnefatafl.TaflBoard()
        start = time.time()
        t = simulate(b.board,
                 b.cache,
                 b.dirty_map,
                 b.dirty_flags,
                 'attackers',
                 visualize=False)
        times.append(time.time() - start)
        turns.append(t)
    print(f"Average rollout time: {sum(times) / len(times)}")
    print(f"Average rollout turns: {sum(turns) / len(turns)}")

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
