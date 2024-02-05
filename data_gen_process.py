import hnefatafl
from simulation import simulate
import random
import sys
import curriculum_learning
from models import FeatureExtractionBlock, AttentionBlock, PolicyHead, ValueHead, PPOViking
import logging

ui = sys.argv[1]
winners = []
logging.basicConfig(filename='curriculum_learning1.log', level=logging.INFO)
for i in range(250):
    if i % 10 == 0:
        print(f"Worker {ui} has finished {i} games...")
    curriculum_choice = random.random()
    if True:
        b = curriculum_learning.generate_attacker_curriculum(1, 1, 5)
        winner = simulate(b.board,
                          b.cache,
                          b.dirty_map,
                          b.dirty_flags,
                          'attackers',
                          b.piece_flags,
                          record=True,
                          visualize=False,
                          show_cache=False,
                          show_dirty=False)
    else:
        b = curriculum_learning.generate_defender_curriculum(1, 5, 5)
        winner = simulate(b.board,
                          b.cache,
                          b.dirty_map,
                          b.dirty_flags,
                          'defenders',
                          b.piece_flags,
                          record=True,
                          visualize=False,
                          show_cache=False,
                          show_dirty=False)
    winners.append(winner)
logging.info(f"Winrate for {ui}: {sum(winners)/len(winners)}")
