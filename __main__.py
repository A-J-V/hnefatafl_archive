import graphics
import hnefatafl
import pygame
import time
import torch
import models
from game_logic import *
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
    b = hnefatafl.initialize_game()
    simulate(b.board,
             b.cache,
             b.dirty_map,
             b.dirty_flags,
             'attackers',
             b.piece_flags,
             record=False,
             visualize=True,
             show_cache=False,
             show_dirty=False)

