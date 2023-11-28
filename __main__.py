import graphics
import hnefatafl
import pygame
import time
from utilities import *
from MCTS import simulate
import cProfile
import random
from MCTS import MCTS
import pandas as pd
import os
import shutil
from models import ResNet_initial_block, ResNet_basic_block, ResNet_transition_block, BabyViking


from testing import run_tests
from test_boards import test_boards as tb


if __name__ == "__main__":
    # start_time = time.time()
    b = hnefatafl.initialize_game()
    #player = "attackers"
    #display = graphics.initialize()
    #graphics.refresh(b.board, display)
    # i = 1
    # while True:
    #     if player == "defenders":
    #       action = input("Type your moves: move row col")
    #       action = [int(i) for i in action.split()]
    #       move, row, col = action
    #     else:
    #         mcts = MCTS(board=b.board,
    #                     cache=b.cache,
    #                     dirty_map=b.dirty_map,
    #                     dirty_flags=b.dirty_flags,
    #                     player=player,
    #                     piece_flags=b.piece_flags,
    #                     max_iter=1)
    #
    #         b = mcts.run()
    #  #   persistent_visits = [child.visits if not isinstance(child, tuple) else 0 for child in b.children]
    #  #   print(f"Persistent visits held over from last turn:\n{persistent_visits}\n")
    #     terminal = is_terminal(b.board, b.cache, b.dirty_map, b.dirty_flags, player, b.piece_flags)
    #     if terminal:
    #         print(f"{terminal} win!")
    #         break
    #     if player == "attackers":
    #         player = "defenders"
    #     else:
    #         player = "attackers"
    #     i += 1
    # print(f"Game took {time.time() - start_time} seconds.")
    #    graphics.refresh(b.board, display)
    # player = "attackers"
    # mcts = MCTS(board=b.board,
    #          cache=b.cache,
    #          dirty_map=b.dirty_map,
    #          dirty_flags=b.dirty_flags,
    #          player=player,
    #          piece_flags=b.piece_flags,
    #          max_iter=100)
    # cProfile.run("mcts.run()")
    # mcts_result = mcts.run()

    # simulate(b.board,
    #         b.cache,
    #         b.dirty_map,
    #         b.dirty_flags,
    #         'attackers',
    #         b.piece_flags,
    #         visualize=True,
    #         show_cache=False,
    #         show_dirty=True)
    #print(b.dirty_flags)
    #cProfile.run("simulate(b.board, b.cache, b.dirty_map, b.dirty_flags, 'attackers', piece_flags=b.piece_flags, visualize=False)")
    # times = []
    # wins = []

    pending_folder = "./pending"
    defender_folder = "/home/alexander/Data/hnefatafl_data/heuristic_train/defender_wins"
    attacker_folder = "/home/alexander/Data/hnefatafl_data/heuristic_train/attacker_wins"

    for i in range(1):
        if i % 50 == 0:
            print(f"{i} games played...")
        b = hnefatafl.TaflBoard()
        start = time.time()
        v = simulate(b.board,
                     b.cache,
                     b.dirty_map,
                     b.dirty_flags,
                     'attackers',
                     b.piece_flags,
                     visualize=False,
                     record=False,
                     snapshot=True)
        #times.append(time.time() - start)
        #v = 1 if v == "defenders" else 0
        #wins.append(v)

        #Move the pending game state snapshots to the correct folder depending on who won
        if v == 1:
            destination_folder = defender_folder
        else:
            destination_folder = attacker_folder
        for filename in os.listdir(pending_folder):
            src_path = os.path.join(pending_folder, filename)
            dest_path = os.path.join(destination_folder, filename)
            shutil.move(src_path, dest_path)
    # print(f"Average rollout time: {sum(times) / len(times)}")
    # print(f"Defender winrate: {sum(wins) / len(wins)}")
    #
    #run_tests()

