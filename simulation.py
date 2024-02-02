import numpy as np

import models
from game_logic import *
import random
import graphics
import time
import pandas as pd
import torch
from datetime import datetime
import logging


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


class MCTS:
    def __init__(self,
                 board: np.array,
                 cache: np.array,
                 dirty_map: dict,
                 dirty_flags: set,
                 player: str,
                 piece_flags: np.array,
                 max_iter: int,
                 device: str,
                 model):
        self.device = device
        self.model = model
        self.caller = player
        self.root_node = Node(board=board,
                              cache=cache,
                              dirty_map=dirty_map,
                              dirty_flags=dirty_flags,
                              player=player,
                              piece_flags=piece_flags)
        self.max_iter = max_iter
        self.iteration = 0

    def run(self):
        """
        Run MCTS to select a move to make.

        :return: A Node object, which is the result of the 'best' move according to MCTS's results.
        """
        while self.iteration < self.max_iter:
            self.iterate()
            self.iteration += 1
        print(f"Total nodes instantiated: {Node.node_count}.")
        expanded_children = sum([1 if not isinstance(child, tuple) else 0 for child in self.root_node.children])
        print(f"Expanded number of children this turn: {expanded_children}")

        Node.node_count = 0
        best_child = self.root_node.get_best_child()
        # Need to implement a selection policy. Currently, the neural version of MCTS
        # Never explores, it always selects the "best" child according to the MCTS exploration.
        print([child.visits for child in self.root_node.children])
        return best_child

    def iterate(self):
        """Make a single iteration of MCTS, from selection through backpropagation."""

        # 1) Selection
        #print("Selection...")
        node = self.root_node
        while not node.terminal and node.is_fully_expanded:
            node = node.select_node()
        #print("Selected.")

        # 2) Expansion
        #print("Expansion...")
        # Depending on implementation, vanilla MCTS may expand one child at a time or all children depending on
        # use-case. Currently, we are only expanding one child to save CPU expense. However, we need to consider
        # if this is most efficient after we switch to AlphaZero style MCTS.
        # Should we instead expand all children so that we can run batch-inference on all of them in the next step?
        #print(f"Node terminal: {node.terminal}")
        #print(f"Node fully expanded: {node.is_fully_expanded}")
        if not node.terminal and not node.is_fully_expanded:
            node = node.expand_child()
            #print("Expanded.")

        # 3) Simulation
        # result = simulate(board=np.array(node.board),
        #                   cache=np.array(node.cache),
        #                   dirty_map=node.dirty_map.copy(),
        #                   dirty_flags=node.dirty_flags.copy(),
        #                   player=node.player,
        #                   piece_flags=np.array(node.piece_flags))
        # if result == self.caller:
        #     result = 1
        # else:
        #     result = 0

        t_board = torch.Tensor(np.array(node.board)).to(self.device).unsqueeze(0)
        with torch.inference_mode():
            pred = self.model(t_board)

        # Should we backpropagate the thresholded classification or the probability?
        result = pred.item()
        if self.caller == "defenders":
            result = result
        else:
            result = 1 - result

        # 4) Backpropagation
        #print("Backpropagation...")
        node.backpropagate(result)
        #print("Backpropagated.")


def toggle_player(player):
    """Returns whichever player isn't the current player."""
    return "defenders" if player == "attackers" else "attackers"


def ucb1(node, c: float = 1.0):
    """Calculate the UCB1 value using exploration factor c."""
    if isinstance(node, tuple) or node.visits == 0:
        return float('inf')
    return ((node.value / (node.visits if node.visits != 0 else 1)) +
            c * (np.log(node.parent.visits) / (node.visits if node.visits != 0 else 1)) ** 0.5
            )


def argmax(lst: list):
    """
    This is returns argmax with ties broken randomly.

    :param lst: List of action scores.
    :return: The argmax of the list of action scores with ties broken randomly.
    """
    if not lst:
        raise Exception("argmax was passed an empty list.")
    max_value = max(lst)
    ties = []
    for i, value in enumerate(lst):
        if value == max_value:
            ties.append(i)
    return random.choice(ties)


def simulate(board: np.array,
             cache: np.array,
             dirty_map: dict,
             dirty_flags: set,
             player: str,
             piece_flags: np.array,
             visualize: bool = False,
             record: bool = False,
             show_cache: bool = False,
             show_dirty: bool = False,
             ):
    """Play through a game on the given board until termination and return the result."""

    if record:
        game_states = []
        game_moves = []
        game_action_space = []
        turn = []

    if visualize:
        display = graphics.initialize()
        graphics.refresh(board, display, piece_flags, show_cache, dirty_flags=dirty_flags, show_dirty=show_dirty)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = models.load_ai()
    model.eval()

    # Add a simple integer cache of how many legal moves each player had last turn.
    attacker_moves = 100
    defender_moves = 100
    turn_num = 1

    while True:

        # Check for termination
        terminal, reason = is_terminal(board=board, cache=cache, dirty_map=dirty_map, dirty_flags=dirty_flags,
                                       player=player,
                                       attacker_moves=attacker_moves, defender_moves=defender_moves,
                                       piece_flags=piece_flags)
        if terminal != 'n/a':
            print(f"{terminal} wins because {reason}.")
            if record:
                game_state_df = pd.DataFrame(game_states)
                game_moves_df = pd.DataFrame(game_moves)
                game_action_space_df = pd.DataFrame(game_action_space)
                game_state_df.columns = ['c' + str(i) for i in list(game_state_df.columns)]
                game_moves_df.columns = ['a_index', 'a_prob']
                game_action_space_df.columns = ['a' + str(i) for i in list(game_action_space_df.columns)]
                game_df = pd.concat([game_state_df, game_moves_df, game_action_space_df], axis=1)
                game_df['winner'] = 1 if terminal == "attackers" else 0
                game_df['turn'] = turn
                game_df.reset_index()
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S_%f")
                game_df.to_csv("./game_recordings/record_" + timestamp + ".csv", index=False)
            return terminal

        # Get a masked action space of legal moves for the player, then get a list of those moves.
        actions = all_legal_moves(board=board, cache=cache, dirty_map=dirty_map,
                                  dirty_flags=dirty_flags, player=player, piece_flags=piece_flags)

        policy_pred, value_pred = model.pred_probs(torch.from_numpy(board).float().unsqueeze(0).to(device),
                                                   player_tensor=get_player_tensor(player).to(device),
                                                   mask=torch.from_numpy(actions).view(-1).to(device))

        #assert torch.allclose(policy_pred, policy_prob), "The policy probability from different sources are not equal"

        # Stochastic action selection
        action_selection = torch.multinomial(policy_pred, 1)
        action_prob = policy_pred[0, action_selection.item()]
        move, row, col = np.unravel_index(action_selection.item(), (40, 11, 11))

        if record:
            flat_board = collapse_board(board)
            flat_actions = collapse_action_space(actions.astype('int'))
            game_states.append(flat_board)
            game_moves.append(np.array([action_selection.item(), action_prob.item()]))
            game_action_space.append(flat_actions)
            turn.append(1 if player == "attackers" else 0)

        actions = np.argwhere(actions == 1)
        # Update the integer cache of legal moves for the current player.
        if player == "attackers":
            attacker_moves = len(actions)
            if attacker_moves == 0:
                return "defenders"
        else:
            defender_moves = len(actions)
            if defender_moves == 0:
                return "attackers"

        new_index = make_move(board,
                              (row, col),
                              move,
                              cache=cache,
                              dirty_map=dirty_map,
                              dirty_flags=dirty_flags,
                              piece_flags=piece_flags)

        # Check for captures around the move
        check_capture(board,
                      new_index,
                      piece_flags=piece_flags,
                      dirty_map=dirty_map,
                      dirty_flags=dirty_flags,
                      cache=cache)

        # Flip the player for the next turn
        player = toggle_player(player)

        if visualize:
            graphics.refresh(board, display, piece_flags, show_cache, dirty_flags=dirty_flags, show_dirty=show_dirty)
            time.sleep(1)
        turn_num += 1


def get_player_tensor(player):
    if player == "attackers":
        player_tensor = torch.ones(1, 11, 11).float().unsqueeze(0)
    else:
        player_tensor = torch.zeros(1, 11, 11).float().unsqueeze(0)
    return player_tensor


# def get_network_probs(board, player, model, actions, device='cuda'):
#
#     board_tensor = torch.from_numpy(board).float().unsqueeze(0).to(device)
#     player_tensor = get_player_tensor(player).to(device)
#
#     with torch.inference_mode():
#         policy_pred, value_pred = model(board_tensor, player_tensor)
#
#     policy_pred = policy_pred.squeeze(0)
#     policy_pred = torch.nn.functional.softmax(policy_pred.view(-1))
#
#     policy_pred = torch.where(torch.from_numpy(actions).view(-1).to(device) == 1,
#                               policy_pred,
#                               torch.zeros_like(policy_pred))
#
#     policy_pred = policy_pred / torch.sum(policy_pred)
#     value_pred = torch.sigmoid(value_pred)
#
#     return policy_pred, value_pred


def heuristic_evaluation(board,
                         cache,
                         dirty_map,
                         dirty_flags,
                         player,
                         defender_moves,
                         attacker_moves,
                         piece_flags,
                         new_index,
                         ):
    # Check thin captures
    captures = check_capture(board,
                             new_index,
                             piece_flags=piece_flags,
                             thin_capture=True,
                             dirty_map=dirty_map,
                             dirty_flags=dirty_flags,
                             cache=cache)

    # Extract features
    features = extract_features(board,
                                defender_moves=defender_moves,
                                attacker_moves=attacker_moves,
                                piece_flags=piece_flags)
    piece_vulnerable = is_vulnerable(board, new_index)

    term, _ = is_terminal(board=board, cache=cache, dirty_map=dirty_map, dirty_flags=dirty_flags,
                          player=player,
                          attacker_moves=attacker_moves, defender_moves=defender_moves,
                          piece_flags=piece_flags)

    if term == 'attackers':
        term = - 1000
    elif term == 'defenders':
        term = 1000
    else:
        term = 0

    # Adjust material count for captures
    if player == 'defenders':
        features['material_balance'] += captures
    elif player == 'attackers':
        features['material_balance'] -= captures

    # Calculate the heuristic value score and assign it to this action
    king_boxed_in = 1 if features['close_defenders'] == 4 else 0
    value = (1.5 * features['material_balance'] - 2 * features['king_dist'] - 1.5 * features['close_attackers']
             - 0.25 * features['attack_options'] + features['escorts'] + 0.2 * features['map_control']
             - king_boxed_in + 12 * features['king_escape'] + features['king_edge'] +
             0.5 * features['king_edge'] * features['close_defenders'] + term)

    if player == 'attackers':
        value = value * -1
    value -= 1.5 * piece_vulnerable
    return value, captures
