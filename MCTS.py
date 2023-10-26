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


def toggle_player(player):
    return "defenders" if player == "attackers" else "attackers"


def best_child(node):
    return max(node.children, key=lambda x: x.value / (x.visits if x.visits != 0 else 1))


def ucb1(node, c):
    """Calculate the UCB1 value using exploration factor c."""
    return ((node.value / (node.visits if node.visits != 0 else 1)) +
            c * (np.log(node.parent.visits) / (node.visits if node.visits != 0 else 1)) ** 0.5
            )


def select_node(node):
    """Use the UCB1 formula to select a node"""
    best = max(node.children, key=ucb1)
    return best if best else None


def expand_node(node):
    actions = all_legal_moves(node.state, node.player)
    for row in actions.shape[1]:
        for col in actions.shape[2]:
            for action in actions[:, row, col]:
                if action == 0:
                    continue
                new_state = np.array(node.state)
                make_move(new_state, (row, col), action)
                new_node = Node(new_state, player=toggle_player(node.player))
                node.children.append(new_node)


def simulate(board, player, visualize=False):
    """Play through a random game on the given board until termination and return the result."""
    if visualize:
        display = graphics.initialize()
        graphics.refresh(board, display)
    i = 0
    while True:
        # Get a masked action space of legal moves for the player
        actions = all_legal_moves(board, player)
        # Get a list of these moves
        actions = np.argwhere(actions == 1)
        # Randomly select one
        choice = random.randint(0, len(actions) - 1)
        move, row, col = actions[choice]
        # Make the action
        new_index = make_move(board, (row, col), move)
        check_capture(board, tuple(new_index))
        i += 1
        if visualize:
            graphics.refresh(board, display)
            time.sleep(0.25)
        # Check for termination
        terminal = is_terminal(board, player)
        if terminal:
            print(terminal)
            print(f"This simulation took {i} turns.")
            break
        else:
            player = toggle_player(player)


class MCTS:

    def __init__(self, initial_state: np.array, player, iterations):
        """Setup for MCTS"""
        self.root_node = Node(state=initial_state, player=player)
        self.iterations = iterations


