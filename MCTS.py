from utilities import *
import random
import graphics
import time
from copy import deepcopy


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
            # Update the new node's state by carrying out the selected move on the copied state
            make_move(board=self.board,
                      index=(spawning_action[1], spawning_action[2]),
                      move=spawning_action[0],
                      cache=self.cache,
                      dirty_map=self.dirty_map,
                      dirty_flags=self.dirty_flags,
                      piece_flags=self.piece_flags)

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
        best_i = 0
        best_value = -float('inf')
        best_child = None
        for i, child in enumerate(self.children):
            value = ucb1(child)
            if value > best_value:
                best_value = value
                best_i = i
                best_child = child
        return best_child

    def expand_children(self):
        for action in self.actions:
            self.lazy_expand_child(action)
        self.is_fully_expanded = True

    def lazy_expand_child(self, action):
        self.children.append(action)

    def expand_child(self):
        # Instantiate the new node with the acted on state and add it as a child node
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
        self.children.append(new_node)
        if not self.actions:
            self.is_fully_expanded = True
        return new_node

    def get_best_child(self):
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
        """Setup for MCTS"""
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
    return "defenders" if player == "attackers" else "attackers"


def ucb1(node, c: float = 1):
    """Calculate the UCB1 value using exploration factor c."""
    if isinstance(node, tuple) or node.visits == 0:
        return float('inf')
    return ((node.value / (node.visits if node.visits != 0 else 1)) +
            c * (np.log(node.parent.visits) / (node.visits if node.visits != 0 else 1)) ** 0.5
            )


def simulate(board: np.array,
             cache: np.array,
             dirty_map: dict,
             dirty_flags: set,
             player: str,
             piece_flags: np.array,
             visualize: bool = False,
             ):
    """Play through a random game on the given board until termination and return the result."""
    if visualize:
        display = graphics.initialize()
        graphics.refresh(board, display)

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
            return terminal
        elif (player == "attackers" and
              quiescent_attacker(board=board, piece_flags=piece_flags)):
            return "attackers"
        elif (player == "defenders" and
              quiescent_defender(board=board,
                                 cache=cache,
                                 dirty_map=dirty_map,
                                 dirty_flags=dirty_flags,
                                 piece_flags=piece_flags)):
            return "defenders"

        # Get a masked action space of legal moves for the player, then get a list of those moves.
        actions = all_legal_moves(board=board, cache=cache, dirty_map=dirty_map,
                                  dirty_flags=dirty_flags, player=player, piece_flags=piece_flags)
        actions = np.argwhere(actions == 1)

        # Update the integer cache of legal moves for the current player.
        if player == "attackers":
            attacker_moves = len(actions)
        else:
            defender_moves = len(actions)

        # Randomly select a legal move and make that move.
        choice = random.randint(0, len(actions) - 1)
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
            graphics.refresh(board, display)
            time.sleep(1)
