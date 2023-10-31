from utilities import *
import random
import graphics
import time


class Node:
    def __init__(self, state: np.array, player, parent=None):
        self.state = state
        self.player = player
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def best_child(self):
        return max(self.children, key=lambda x: x.value / (x.visits if x.visits != 0 else 1))

    def ucb1(self, c):
        """Calculate the UCB1 value using exploration factor c."""
        return ((self.value / (self.visits if self.visits != 0 else 1)) +
                c * (np.log(self.parent.visits) / (self.visits if self.visits != 0 else 1)) ** 0.5
                )

    def select_node(self):
        """Use the UCB1 formula to select a node"""
        best = max(self.children, key=self.ucb1)
        return best if best else None

    def expand_node(self):
        actions = all_legal_moves(self.state, self.player)
        for row in actions.shape[1]:
            for col in actions.shape[2]:
                for action in actions[:, row, col]:
                    if action == 0:
                        continue
                    new_state = np.array(self.state)
                    make_move(new_state, (row, col), action)
                    new_node = Node(new_state, player=toggle_player(self.player))
                    self.children.append(new_node)


def toggle_player(player):
    return "defenders" if player == "attackers" else "attackers"


def simulate(board,
             cache,
             dirty_map,
             dirty_flags,
             player,
             visualize=False):
    """Play through a random game on the given board until termination and return the result."""
    if visualize:
        display = graphics.initialize()
        graphics.refresh(board, display)
    i = 0
    # Add a version simple integer cache of how many legal moves each player had last turn
    attacker_moves = 100
    defender_moves = 100
    while True:
        # Get a masked action space of legal moves for the player
        actions = all_legal_moves(board=board, cache=cache, dirty_map=dirty_map, dirty_flags=dirty_flags, player=player)
        # Get a list of these moves
        actions = np.argwhere(actions == 1)
        if player == "attackers":
            attacker_moves = len(actions)
        else:
            defender_moves = len(actions)

        # Randomly select one
        choice = random.randint(0, len(actions) - 1)
        move, row, col = actions[choice]
        # Make the action
        new_index = make_move(board, (row, col), move, cache=cache, dirty_map=dirty_map, dirty_flags=dirty_flags)
        check_capture(board, new_index)
        i += 1
        if visualize:
            graphics.refresh(board, display)
            time.sleep(1)
        # Check for termination
        terminal = is_terminal(board=board, cache=cache, dirty_map=dirty_map, dirty_flags=dirty_flags, player=player,
                               attacker_moves=attacker_moves, defender_moves=defender_moves)
        if terminal:
            #print(terminal)
            #print(f"This simulation took {i} turns.")
            return i
            #break
        else:
            if (player == "defenders" and
               quiescent_defender(board=board, cache=cache, dirty_map=dirty_map, dirty_flags=dirty_flags)):
                return i
            elif (player == "attackers" and
                  quiescent_attacker(board=board, cache=cache, dirty_map=dirty_map, dirty_flags=dirty_flags)):
                return i
            player = toggle_player(player)


class MCTS:

    def __init__(self, initial_state: np.array, player, iterations):
        """Setup for MCTS"""
        self.root_node = Node(state=initial_state, player=player)
        self.iterations = iterations


