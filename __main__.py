import time
import torch

import hnefatafl
import models
from game_logic import *
from models import FeatureExtractionBlock, AttentionBlock, PolicyHead, ValueHead, PPOViking
from simulation import simulate
import curriculum_learning

import train
import subprocess
import os
import logging


from testing import run_tests
from test_boards import test_boards as tb


if __name__ == "__main__":

    # b = hnefatafl.initialize_game()
    # simulate(board=b.board,
    #          cache=b.cache,
    #          dirty_map=b.dirty_map,
    #          dirty_flags=b.dirty_flags,
    #          player="attackers",
    #          piece_flags=b.piece_flags,
    #          visualize=True,
    #          record=False)
    logging.basicConfig(filename='curriculum_learning1.log', level=logging.INFO)
    for iteration in range(200):
        start = time.time()
        # subprocess.run(["./hnefatafl_generate.sh"], check=False)
        # time.sleep(30)

        train.training_iteration(model="NV_attacker",
                                 player=1,
                                 batch_size=512,
                                 weight_decay=0.0002,
                                 c1=0.75,
                                 c2=None,
                                 checkpoint=str(iteration),
                                 )
        train.training_iteration(model="NV_defender",
                                 player=0,
                                 batch_size=512,
                                 weight_decay=0.0002,
                                 c1=0.75,
                                 c2=None,
                                 checkpoint=str(iteration),
                                 )

        time.sleep(5)
        files = os.listdir("./game_recordings")
        for file in files:
            file_path = os.path.join("./game_recordings", file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"Iteration {iteration} took {time.time() - start} seconds.")
