import numpy as np
import models
from game_logic import *
import random
import graphics
import time
import pandas as pd
import torch
from datetime import datetime
from simulation import *


class Node:
    node_count = 0

    def __init__(self,
                 board: np.array,
                 cache: np.array,
                 dirty_map: dict,
                 dirty_flags: set,
                 player: str,
                 piece_flags: np.array,
                 parent=None,
                 spawning_action=None
                 ) -> None:
        self.board = np.array(board)
        self.cache = np.array(cache)
        self.dirty_map = dirty_map.copy()
        self.dirty_flags = dirty_flags.copy()
        self.player = player
        self.piece_flags = np.array(piece_flags)
        self.parent = parent
        self.spawning_action = spawning_action
        self.children = []
        self.visits = 0
        self.value = 0

        if spawning_action:
            # Update the new node's state by carrying out the action that creates it
            new_index = make_move(board=self.board,
                                  index=(spawning_action[1], spawning_action[2]),
                                  move=spawning_action[0],
                                  cache=self.cache,
                                  dirty_map=self.dirty_map,
                                  dirty_flags=self.dirty_flags,
                                  piece_flags=self.piece_flags)

            # Check for captures around the move
            check_capture(self.board,
                          new_index,
                          piece_flags=self.piece_flags,
                          dirty_map=dirty_map,
                          dirty_flags=dirty_flags,
                          cache=cache)

        # Get the legal actions that can be done in this Node's state
        actions = all_legal_moves(board=self.board,
                                  cache=self.cache,
                                  dirty_map=self.dirty_map,
                                  dirty_flags=self.dirty_flags,
                                  player=self.player,
                                  piece_flags=self.piece_flags
                                  )
        actions = np.argwhere(actions == 1)
        # This isn't guaranteed to be a random order
        self.actions = {(move, row, col) for move, row, col in actions}

        # Check whether this Node is terminal
        self.winner = is_terminal(board=self.board,
                                  cache=self.cache,
                                  dirty_map=self.dirty_map,
                                  dirty_flags=self.dirty_flags,
                                  player=self.player,
                                  piece_flags=self.piece_flags)
        #print(self.winner)
        self.terminal = False if self.winner == ('n/a', 'n/a') else True

        # Store whether this Node is fully expanded
        self.is_fully_expanded = False

    def select_node(self):
        """Use the UCB1 formula to select a node"""
        best_value = -float('inf')
        best_child = None
        for child in self.children:
            value = ucb1(child)
            if value > best_value:
                best_value = value
                best_child = child
        return best_child

    def expand_child(self):
        """Take an action from the Node's list of actions, and instantiate a new Node with that action."""

        # Pop one of this Node's actions. Create a new Node, and pass in that action as the spawning action.
        action = self.actions.pop()
        new_node = Node(board=self.board,
                        cache=self.cache,
                        dirty_map=self.dirty_map,
                        dirty_flags=self.dirty_flags,
                        player=toggle_player(self.player),
                        piece_flags=self.piece_flags,
                        parent=self,
                        spawning_action=action)
        Node.node_count += 1

        # Append the new Node to the list of this Node's children.
        self.children.append(new_node)

        # If that was the last unexplored action of this Node, set this Node as fully expanded.
        if not self.actions:
            self.is_fully_expanded = True
        return new_node

    def get_best_child(self):
        """Return the 'best' child of this Node according to number of visits."""
        visit_counts = [child.visits for child in self.children]
        max_visit_index = argmax(visit_counts)
        best_child = self.children[max_visit_index]
        return best_child

    def backpropagate(self, result):
        """Backpropagate the result of exploration up the game tree."""
        self.visits += 1
        self.value += result
        if self.parent:
            self.parent.backpropagate(1 if result == 0 else 0)