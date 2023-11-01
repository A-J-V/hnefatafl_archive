from utilities import *
import random
import graphics
import time


class Node:
    def __init__(self,
                 board: np.array,
                 cache: np.array,
                 dirty_map: dict,
                 dirty_flags: set,
                 player: str,
                 parent=None,
                 ) -> None:
        self.board = np.array(board)
        self.cache = np.array(cache)
        self.dirty_map = dirty_map.copy()
        self.dirty_flags = dirty_flags.copy()
        self.player = player
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        actions = all_legal_moves(board=self.board,
                                  cache=self.cache,
                                  dirty_map=self.dirty_map,
                                  dirty_flags=self.dirty_flags,
                                  player=self.player,
                                  )
        actions = np.argwhere(actions == 1)
        self.actions = [(move, row, col) for move, row, col in actions]
        random.shuffle(self.actions)

    def best_child(self):
        return max(self.children, key=lambda x: x.value / (x.visits if x.visits != 0 else 1))

    def ucb1(self, c: float = 1.41):
        """Calculate the UCB1 value using exploration factor c."""
        return ((self.value / (self.visits if self.visits != 0 else 1)) +
                c * (np.log(self.parent.visits) / (self.visits if self.visits != 0 else 1)) ** 0.5
                )

    def select_node(self):
        """Use the UCB1 formula to select a node"""
        best = max(self.children, key=self.ucb1)
        return best if best else None

    def expand_node(self):
        actions = all_legal_moves(board=self.board,
                                  cache=self.cache,
                                  dirty_map=self.dirty_map,
                                  dirty_flags=self.dirty_flags,
                                  player=self.player,
                                  )
        move, row, col = self.actions.pop()
        new_state = np.array(self.board)
        # Needs to be updated to reflect correct game mechanics
        make_move(new_state, (row, col), move)
        new_node = Node(new_state, player=toggle_player(self.player))
        self.children.append(new_node)


def toggle_player(player):
    return "defenders" if player == "attackers" else "attackers"


def simulate(board: np.array,
             cache: np.array,
             dirty_map: dict,
             dirty_flags: set,
             player: str,
             visualize: bool = False,  # This can be removed after dev and debugging is finished
             ):
    """Play through a random game on the given board until termination and return the result."""
    if visualize:
        display = graphics.initialize()
        graphics.refresh(board, display)

    # For debugging, record number of turns in the simulation i
    i = 0

    # Add a simple integer cache of how many legal moves each player had last turn.
    attacker_moves = 100
    defender_moves = 100
    while True:

        # Get a masked action space of legal moves for the player, then get a list of those moves.
        actions = all_legal_moves(board=board, cache=cache, dirty_map=dirty_map, dirty_flags=dirty_flags, player=player)
        actions = np.argwhere(actions == 1)

        # Update the integer cache of legal moves for the current player.
        if player == "attackers":
            attacker_moves = len(actions)
        else:
            defender_moves = len(actions)

        # Randomly select a legal move and make that move.
        choice = random.randint(0, len(actions) - 1)
        move, row, col = actions[choice]
        new_index = make_move(board, (row, col), move, cache=cache, dirty_map=dirty_map, dirty_flags=dirty_flags)

        # Check for captures around the move
        check_capture(board, new_index)

        # Check for termination
        terminal = is_terminal(board=board, cache=cache, dirty_map=dirty_map, dirty_flags=dirty_flags, player=player,
                               attacker_moves=attacker_moves, defender_moves=defender_moves)
        if terminal:
            win = 1 if player == "defenders" else 0
            return i, win
        else:
            if (player == "defenders" and
               quiescent_attacker(board=board)):
                return i, 0
            elif (player == "attackers" and
                  quiescent_defender(board=board, cache=cache, dirty_map=dirty_map, dirty_flags=dirty_flags)):
                return i, 1
            player = toggle_player(player)

        # For debugging
        i += 1
        if visualize:
            graphics.refresh(board, display)
            time.sleep(1)


class MCTS:

    def __init__(self, initial_state: np.array, player, iterations):
        """Setup for MCTS"""
        self.root_node = Node(board=initial_state, player=player)
        self.iterations = iterations


