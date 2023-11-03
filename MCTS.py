from utilities import *
import random
import graphics
import time
from copy import deepcopy


class Node:
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
        self.actions = [(move, row, col) for move, row, col in actions]
        random.shuffle(self.actions)

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
        # If we only expand one child at a time, this needs to be modified so that we have a chance of expanding an
        # unexplored child node.
        best = max(self.children, key=ucb1)
        return best if best else None

    def expand_children(self):
        #print("Inside expand_children()...")
        for action in self.actions:
            self.expand_node(action)
        self.is_fully_expanded = True
        return random.choice(self.children)

    def expand_node(self, action):
        # Instantiate the new node with the acted on state and add it as a child node
        new_node = Node(board=self.board,
                        cache=self.cache,
                        dirty_map=self.dirty_map,
                        dirty_flags=self.dirty_flags,
                        player=toggle_player(self.player),
                        piece_flags=self.piece_flags,
                        parent=self,
                        spawning_action=action)
        self.children.append(new_node)

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
            #print(f"Running iteration {self.iteration}...")
            self.iterate()
            self.iteration += 1

        return max(self.root_node.children, key=lambda x: x.visits).spawning_action

    def iterate(self):

        # 1) Selection
        node = self.root_node
        #print(f"node.terminal: {node.terminal}.")
        #print(f"node.is_fully_expanded: {node.is_fully_expanded}.")
        while not node.terminal and node.is_fully_expanded:
            #print(f"Selecting node...")
            node = node.select_node()

        # 2) Expansion
        if not node.terminal and not node.is_fully_expanded:
            #print(f"Expanding children...")
            node = node.expand_children()

        # 3) Simulation
        #print(f"Running simulation...")
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
        #print("Backpropagating...")
        node.backpropagate(result)


def toggle_player(player):
    return "defenders" if player == "attackers" else "attackers"


def ucb1(node, c: float = 1.5):
    """Calculate the UCB1 value using exploration factor c."""
    return ((node.value / (node.visits if node.visits != 0 else 1)) +
            c * (np.log(node.parent.visits) / (node.visits if node.visits != 0 else 1)) ** 0.5
            )


def simulate(board: np.array,
             cache: np.array,
             dirty_map: dict,
             dirty_flags: set,
             player: str,
             piece_flags: np.array,
             visualize: bool = False,  # This can be removed after dev and debugging is finished
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
