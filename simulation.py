import numpy as np

import models
from game_logic import *
from node import Node
import random
import graphics
import time
import pandas as pd
import torch
from datetime import datetime
import logging


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
        player_tensor = get_player_tensor(self.caller)
        actions = all_legal_moves(board=node.board, cache=node.cache, dirty_map=node.dirty_map,
                                  dirty_flags=node.dirty_flags, player=node.player, piece_flags=node.piece_flags)
        flat_actions = collapse_action_space(actions.astype('int'))
        mask_tensor = torch.tensor(flat_actions)

        if self.caller == "attackers":
            player_tensor += 1

        with torch.inference_mode():
            _, value_pred = self.model(t_board, player_tensor, mask_tensor)

        # Should we backpropagate the thresholded classification or the probability?
        result = value_pred.item()

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
        # Needed for training
        game_states = []
        game_moves = []
        game_action_space = []
        turn = []
        value_estimates = []

    if visualize:
        display = graphics.initialize()
        graphics.refresh(board, display, piece_flags, show_cache, dirty_flags=dirty_flags, show_dirty=show_dirty)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_a = models.load_ai(model="NV_attacker")
    model_b = models.load_ai(model="NV_defender")

    model_a.eval()
    model_b.eval()

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
            # If there turn is 1, then this was an aborted curriculum learning attempt.
            if turn_num == 1:
                return -1
            print(f"{terminal} wins because {reason}.")
            if record:
                game_state_df = pd.DataFrame(game_states)
                game_moves_df = pd.DataFrame(game_moves)
                game_action_space_df = pd.DataFrame(game_action_space)
                game_state_df.columns = ['c' + str(i) for i in list(game_state_df.columns)]
                game_moves_df.columns = ['a_index', 'a_prob']
                game_action_space_df.columns = ['a' + str(i) for i in list(game_action_space_df.columns)]
                game_df = pd.concat([game_state_df, game_moves_df, game_action_space_df], axis=1)
                winner = 1 if terminal == "attackers" else 0
                game_df['winner'] = winner
                game_df['turn'] = turn
                game_df['v_est'] = value_estimates
                game_df['v_est_next'] = game_df['v_est'].shift(-1, fill_value=0)
                game_df['td_error_attacker'] = get_td_error(game_df['v_est'],
                                                            game_df['v_est_next'],
                                                            1,
                                                            winner)
                game_df['td_error_defender'] = get_td_error(game_df['v_est'],
                                                            game_df['v_est_next'],
                                                            0,
                                                            winner)
                game_df['gae_attacker'] = calculate_gae(game_df['td_error_attacker'], lambda_=0.99)
                game_df['gae_defender'] = calculate_gae(game_df['td_error_defender'], lambda_=0.99)
                game_df.reset_index()
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S_%f")
                game_df.to_csv("./game_recordings/record_" + timestamp + ".csv", index=False)

                return winner
            return terminal

        # Get a masked action space of legal moves for the player, then get a list of those moves.
        actions = all_legal_moves(board=board, cache=cache, dirty_map=dirty_map,
                                  dirty_flags=dirty_flags, player=player, piece_flags=piece_flags)

        if player == "attackers":
            model = model_a
        elif player == "defenders":
            model = model_b
        else:
            raise Exception("Unrecognized player.")

        with torch.inference_mode():
            policy_pred, value_pred = model.pred_probs(torch.from_numpy(board).float().unsqueeze(0).to(device),
                                                       player_tensor=get_player_tensor(player).to(device),
                                                       mask=torch.from_numpy(actions).view(-1).to(device))

        # Stochastic action selection
        try:
            action_selection = torch.multinomial(policy_pred, 1)
            action_prob = policy_pred[0, action_selection.item()]
        except:
            print("Encountered error at torch.multinomial... This needs to be fixed!")
            print(f"The policy preds contains nan: {torch.isnan(policy_pred).any()}")
            print(f"The policy preds contains inf: {torch.isinf(policy_pred).any()}")
            print(f"The policy preds contains < 0: {(policy_pred < 0).any()}")
            pred_clone = policy_pred.clone()
            pred_clone = torch.where((torch.isnan(pred_clone)) | (pred_clone == 0), 0, 0.1)
            pred_clone /= torch.sum(pred_clone, dim=1, keepdim=True)
            action_selection = torch.multinomial(pred_clone, 1)
            action_prob = pred_clone[0, action_selection.item()]

        move, row, col = np.unravel_index(action_selection.item(), (40, 11, 11))

        if record:
            flat_board = collapse_board(board)
            flat_actions = collapse_action_space(actions.astype('int'))
            game_states.append(flat_board)
            game_moves.append(np.array([action_selection.item(), action_prob.item()]))
            game_action_space.append(flat_actions)
            turn.append(1 if player == "attackers" else 0)
            value_estimates.append(value_pred.item())

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


def get_td_error(vt, vtp, player, winner):
    """Given the value estimates for t and t+1, player, and winner, calculate TD error for the player."""
    rewards = np.zeros_like(vt.values)
    if winner == player:
        rewards[-1] = 1
    else:
        rewards[-1] = -1

    td_error = vtp - vt + rewards
    return td_error


def calculate_gae(td_error, gamma=0.99, lambda_=0.95):
    # Calculate GAE
    gae = []
    gae_t = 0
    for t in reversed(range(len(td_error))):
        delta = td_error.iloc[t]
        gae_t = delta + gamma * lambda_ * gae_t
        gae.insert(0, gae_t)
    return gae


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
