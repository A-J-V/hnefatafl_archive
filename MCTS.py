from utilities import *
import random
import graphics
import time
import pandas as pd


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
             ):
    """Play through a random game on the given board until termination and return the result."""
    if visualize:
        display = graphics.initialize()
        graphics.refresh(board, display)

    if record:
        df = pd.DataFrame()

    # Add a simple integer cache of how many legal moves each player had last turn.
    attacker_moves = 100
    defender_moves = 100
    while True:

        # Check for termination
        terminal = is_terminal(board=board, cache=cache, dirty_map=dirty_map, dirty_flags=dirty_flags,
                               player=player,
                               attacker_moves=attacker_moves, defender_moves=defender_moves,
                               piece_flags=piece_flags)
        if terminal:
            print(f"{terminal} wins.")
            return terminal
        elif (player == "attackers" and
              quiescent_attacker(board=board, piece_flags=piece_flags)):
            print("Attackers win (quiescent).")
            return "attackers"
        elif (player == "defenders" and
              quiescent_defender(board=board,
                                 cache=cache,
                                 dirty_map=dirty_map,
                                 dirty_flags=dirty_flags,
                                 piece_flags=piece_flags)):
            print("Defenders win (quiescent).")
            return "defenders"

        # Get a masked action space of legal moves for the player, then get a list of those moves.
        actions = all_legal_moves(board=board, cache=cache, dirty_map=dirty_map,
                                  dirty_flags=dirty_flags, player=player, piece_flags=piece_flags)
        actions = np.argwhere(actions == 1)

        # Update the integer cache of legal moves for the current player.
        if player == "attackers":
            attacker_moves = len(actions)
            if attacker_moves == 0:
                print("Attackers have no legal moves!")
                return "defenders"
        else:
            defender_moves = len(actions)
            if defender_moves == 0:
                print("Defenders have no legal moves!")
                return "attackers"

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

            # Check thin captures
            captures = check_capture(board, new_index, piece_flags=piece_flags, thin_capture=True)

            # Extract features
            material_balance, king_dist_to_corner, mobility_delta, escorts, \
            attack_options, map_control, close_defenders, close_attackers, king_escape = extract_features(board,
                                                                                                          defender_moves=defender_moves,
                                                                                                          attacker_moves=attacker_moves,
                                                                                                          thin=False,
                                                                                                          piece_flags=piece_flags)
            piece_vulnerable = is_vulnerable(board, new_index)

            # Revert the temporary move
            revert_move(board, new_index=new_index, old_index=old_index, piece_flags=piece_flags)

            # Adjust material count for captures
            if player == 'defenders':
                material_balance += captures
            elif player == 'attackers':
                material_balance -= captures

            # Calculate the heuristic value score and assign it to this action
            king_boxed_in = 1 if close_defenders == 4 else 0
            value = 1.5 * material_balance - 2 * king_dist_to_corner - 1.5 * close_attackers - 0.25 * attack_options
            value += escorts + 0.20 * map_control - king_boxed_in + 10 * king_escape
            if player == 'attackers':
                value = value * -1
            value -= 1.5 * piece_vulnerable
            action_scores.append(value)

        # Epsilon greedy move selection
        if random.random() < 0.1:
            # Randomly select a legal move and make that move.
            choice = random.randint(0, len(actions) - 1)
            move, row, col = actions[choice]
        else:
            choice = argmax(action_scores)
            move, row, col = actions[choice]

        # # Randomly select a legal move and make that move.
        # choice = random.randint(0, len(actions) - 1)
        # move, row, col = actions[choice]

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
            graphics.refresh(board, display)
            time.sleep(1)

