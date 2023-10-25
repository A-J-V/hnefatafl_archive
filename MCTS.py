from utilities import *


class Node:
    def __init__(self, state: np.array, player, parent=None):
        self.state = state
        self.player = player
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0


def toggle_player(player):
    return "defender" if player == "attacker" else "attacker"


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


def simulate(board, player):
    pass


class MCTS:

    def __init__(self, initial_state: np.array, player, iterations):
        """Setup for MCTS"""
        self.root_node = Node(state=initial_state, player=player)
        self.iterations = iterations


