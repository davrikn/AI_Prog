from configs import size
from typing import TypeVar
from gameWorld import GameWorld
Node = TypeVar("Node", bound="MonteCarloNode")

class MonteCarloNode:
    def __init__(self, game: GameWorld, parent: Node):
        self.parent = parent
        self.state = game


class MonteCarlo:
    def __init__(self, initial_state = GameWorld(size)):
        self.initial_state = MonteCarloNode(initial_state)

    def limited_bfs(self, depth = 3):
        node = self.initial_state
