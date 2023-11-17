import models
from utilities import *
import random
import graphics
import time
import pandas as pd
import torch
from datetime import datetime


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
            check_capture(self.board, new_index, piece_flags=self.piece_flags)

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
        self.terminal = False if self.winner is None else True

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
        max_visits = -1
        best_child = None
        for child in self.children:
            if isinstance(child, tuple):
                continue
            else:
                if child.visits > max_visits:
                    best_child = child
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
                 max_iter: int = 500):
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
        return best_child

    def iterate(self):
        """Make a single iteration of MCTS, from selection through backpropagation."""

        # 1) Selection
        node = self.root_node
        while not node.terminal and node.is_fully_expanded:
            node = node.select_node()

        # 2) Expansion
        if not node.terminal and not node.is_fully_expanded:
            node = node.expand_child()

        # 3) Simulation
        result = simulate(board=np.array(node.board),
                          cache=np.array(node.cache),
                          dirty_map=node.dirty_map.copy(),
                          dirty_flags=node.dirty_flags.copy(),
                          player=node.player,
                          piece_flags=np.array(node.piece_flags))
        if result == self.caller:
            result = 1
        else:
            result = 0

        # 4) Backpropagation
        node.backpropagate(result)


def toggle_player(player):
    """Returns whichever player isn't the current player."""
    return "defenders" if player == "attackers" else "attackers"


def ucb1(node, c: float = 1):
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
             snapshot: bool = False,
             ):
    """Play through a random game on the given board until termination and return the result."""
    if visualize:
        display = graphics.initialize()
        graphics.refresh(board, display, piece_flags)

    df = pd.DataFrame(columns=['material_balance',
                               'king_dist',
                               'escorts',
                               'attack_options',
                               'close_defenders',
                               'close_attackers',
                               'mobility_delta',
                               'map_control',
                               'king_escape',
                               'king_edge',
                               'player',
                               'turn_num',
                               ])

    # Add a simple integer cache of how many legal moves each player had last turn.
    attacker_moves = 100
    defender_moves = 100
    turn_num = 1
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = models.load_baby_viking()
    #
    # board_size = 11
    # special_tiles = torch.zeros((1, board_size, board_size))
    # # Mark corners and center (adjust indices based on your board layout)
    # corner_indices = [0, board_size - 1]
    # for i in corner_indices:
    #     for j in corner_indices:
    #         special_tiles[0, i, j] = 1
    # special_tiles[0, board_size // 2, board_size // 2] = -1  # Center tile
    # special_tiles = special_tiles.to(device).unsqueeze(0)
    while True:

        if record:
            # At start of turn, append the observation to the dataframe
            obs = extract_features(board,
                                   defender_moves=defender_moves,
                                   attacker_moves=attacker_moves,
                                   piece_flags=piece_flags)
            obs['player'] = player
            obs['turn_num'] = turn_num
            df.loc[len(df)] = obs

        if snapshot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            file_name = f"./pending/snapshot_{timestamp}.pt"
            tensor_state = torch.Tensor(board)
            torch.save(tensor_state, file_name)


        # Check for termination
        terminal, reason = is_terminal(board=board, cache=cache, dirty_map=dirty_map, dirty_flags=dirty_flags,
                                       player=player,
                                       attacker_moves=attacker_moves, defender_moves=defender_moves,
                                       piece_flags=piece_flags)
        if terminal != 'n/a':
            print(f"{terminal} wins because {reason}.")
            if record:
                df['final_turn'] = turn_num
                df['victory_condition'] = reason
                df['winner'] = terminal

            return terminal, df
        # elif (player == "attackers" and
        #       quiescent_attacker(board=board, piece_flags=piece_flags)):
        #     print("Attackers win (quiescent).")
        #     if record:
        #         df['final_turn'] = turn_num
        #         df['victory_condition'] = 'quiescent_attackers'
        #         df['winner'] = 'attackers'
        #     return "attackers", df
        # elif (player == "defenders" and
        #       quiescent_defender(board=board,
        #                          cache=cache,
        #                          dirty_map=dirty_map,
        #                          dirty_flags=dirty_flags,
        #                          piece_flags=piece_flags)):
        #     print("Defenders win (quiescent).")
        #     if record:
        #         df['final_turn'] = turn_num
        #         df['victory_condition'] = 'quiescent_defenders'
        #         df['winner'] = 'defenders'
        #     return "defenders", df

        # Get a masked action space of legal moves for the player, then get a list of those moves.
        actions = all_legal_moves(board=board, cache=cache, dirty_map=dirty_map,
                                  dirty_flags=dirty_flags, player=player, piece_flags=piece_flags)
        actions = np.argwhere(actions == 1)

        # Update the integer cache of legal moves for the current player.
        if player == "attackers":
            attacker_moves = len(actions)
            if attacker_moves == 0:
                print("Attackers have no legal moves!")
                df['final_turn'] = turn_num
                df['victory_condition'] = 'attackers_no_moves'
                df['winner'] = 'defenders'
                return "defenders", df
        else:
            defender_moves = len(actions)
            print(f'Defenders have {defender_moves} moves.')
            if defender_moves == 0:
                print("Defenders have no legal moves!")
                df['final_turn'] = turn_num
                df['victory_condition'] = 'defenders_no_moves'
                df['winner'] = 'defenders'
                return "attackers", df

        # Move evaluation logic goes here.
        action_scores = []
        for move, row, col in actions:
            # Make a thin move
            new_index, old_index = make_move(board,
                                             (row, col),
                                             move,
                                             cache=cache,
                                             dirty_map=dirty_map,
                                             dirty_flags=dirty_flags,
                                             piece_flags=piece_flags,
                                             thin_move=True)

            # Neural Net Evaluation
            # t_board = torch.Tensor(board).to(device).unsqueeze(0)
            # t_board = torch.cat((t_board, special_tiles), dim=1)
            # value = model(t_board)
            # if player == 'attackers':
            #    value = 1 - value

            # Heuristic Evaluation
            value = heuristic_evaluation(board=board,
                                         cache=cache,
                                         dirty_map=dirty_map,
                                         dirty_flags=dirty_flags,
                                         player=player,
                                         defender_moves=defender_moves,
                                         attacker_moves=attacker_moves,
                                         piece_flags=piece_flags,
                                         new_index=new_index
                                         )

            # Revert the temporary move
            revert_move(board, new_index=new_index, old_index=old_index, piece_flags=piece_flags)

            action_scores.append(value)

        # Epsilon greedy move selection
        if random.random() < 0.00:
            # Randomly select a legal move and make that move.
            choice = random.randint(0, len(actions) - 1)
            move, row, col = actions[choice]
        else:
            choice = argmax(action_scores)
            move, row, col = actions[choice]

        new_index = make_move(board,
                              (row, col),
                              move,
                              cache=cache,
                              dirty_map=dirty_map,
                              dirty_flags=dirty_flags,
                              piece_flags=piece_flags)

        # Check for captures around the move
        check_capture(board, new_index, piece_flags=piece_flags)

        # Flip the player for the next turn
        player = toggle_player(player)

        if visualize:
            graphics.refresh(board, display, piece_flags)
            time.sleep(1)
        turn_num += 1


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
    captures = check_capture(board, new_index, piece_flags=piece_flags, thin_capture=True)

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
             - king_boxed_in + 10 * features['king_escape'] + features['king_edge'] +
             0.5 * features['king_edge'] * features['close_defenders'] + term)

    if player == 'attackers':
        value = value * -1
    value -= 1.5 * piece_vulnerable
    return value
