import graphics
import hnefatafl
import pygame
import time
import torch
import models
from utils import *
from models import FeatureExtractionBlock, AttentionBlock, PolicyHead, ValueHead, PPOViking
from simulation import simulate
import cProfile
import random
from simulation import MCTS
import pandas as pd
import os
import shutil


from testing import run_tests
from test_boards import test_boards as tb


if __name__ == "__main__":
    # start_time = time.time()
    #b = hnefatafl.initialize_game()
    # flat = collapse_board(b.board)
    # print(flat)
    # raise Exception()
    # player = "attackers"
    # display = graphics.initialize()
    # graphics.refresh(b.board, display, piece_flags=b.piece_flags)
    # i = 1
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = models.load_ai()
    # model.eval()
    # while True:
    #
    #     mcts = MCTS(board=b.board,
    #                 cache=b.cache,
    #                 dirty_map=b.dirty_map,
    #                 dirty_flags=b.dirty_flags,
    #                 player=player,
    #                 piece_flags=b.piece_flags,
    #                 max_iter=500,
    #                 device=device,
    #                 model=model)
    #
    #     b = mcts.run()
    #     persistent_visits = sum([child.visits for child in b.children])
    #     print(f"Persistent visits held over from last turn:\n{persistent_visits}\n")
    #     terminal = is_terminal(b.board, b.cache, b.dirty_map, b.dirty_flags, player, b.piece_flags)
    #     if terminal != ('n/a', 'n/a'):
    #         print(f"{terminal} win!")
    #         break
    #     if player == "attackers":
    #         player = "defenders"
    #     else:
    #         player = "attackers"
    #     i += 1
    #     graphics.refresh(b.board, display, piece_flags=b.piece_flags)
    # print(f"Game took {time.time() - start_time} seconds.")

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
    times = []
    for _ in range(100):
        b = hnefatafl.initialize_game()
        start = time.time()
        simulate(b.board,
                 b.cache,
                 b.dirty_map,
                 b.dirty_flags,
                 'attackers',
                 b.piece_flags,
                 record=False,
                 visualize=False,
                 show_cache=False,
                 show_dirty=False,
                 neural=True)
        times.append(time.time() - start)
    print(times)
    print(f"Average time per game: {sum(times) / len(times)}")
    #print(b.dirty_flags)
    #cProfile.run("simulate(b.board, b.cache, b.dirty_map, b.dirty_flags, 'attackers', piece_flags=b.piece_flags, visualize=False)")
    # times = []
    # wins = []

    # pending_folder = "./pending"
    # defender_folder = "/home/alexander/Data/hnefatafl_data/heuristic_train/defender_wins"
    # attacker_folder = "/home/alexander/Data/hnefatafl_data/heuristic_train/attacker_wins"
    #
    # for i in range(1):
    #     if i % 50 == 0:
    #         print(f"{i} games played...")
    #     b = hnefatafl.TaflBoard()
    #     start = time.time()
    #     v = simulate(b.board,
    #                  b.cache,
    #                  b.dirty_map,
    #                  b.dirty_flags,
    #                  'attackers',
    #                  b.piece_flags,
    #                  visualize=False,
    #                  record=False,
    #                  snapshot=True)
        #times.append(time.time() - start)
        #v = 1 if v == "defenders" else 0
        #wins.append(v)

        #Move the pending game state snapshots to the correct folder depending on who won
        # if v == 1:
        #     destination_folder = defender_folder
        # else:
        #     destination_folder = attacker_folder
        # for filename in os.listdir(pending_folder):
        #     src_path = os.path.join(pending_folder, filename)
        #     dest_path = os.path.join(destination_folder, filename)
        #     shutil.move(src_path, dest_path)
    # print(f"Average rollout time: {sum(times) / len(times)}")
    # print(f"Defender winrate: {sum(wins) / len(wins)}")
    #
    #run_tests()

