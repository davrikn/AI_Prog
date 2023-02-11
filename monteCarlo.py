from configs import size
from typing import TypeVar
from gameWorld import GameWorld
import numpy as np

Node = TypeVar("Node", bound="MonteCarloNode")


class MonteCarloNode:
    def __init__(self, game: GameWorld, parent: Node):
        self.parent = parent
        self.state = game
        self.number_of_visits = 0
        self.edge_traversals = {}


class MonteCarlo:
    def __init__(self, initial_state=GameWorld(size)):
        self.initial_state = MonteCarloNode(initial_state)

    def limited_bfs(self, depth=3):
        node = self.initial_state

    def select_node_to_expand(self, leaf_nodes: list[MonteCarloNode]) -> MonteCarloNode:
        """Return a Node

        Uses the Upper Confidence Bounds (UCB) to select which leaf node to expand
        """

        # TODO: fant ikke en oneliner for dette, HJELP DAVID
        node_to_expand = leaf_nodes[0]
        for node in leaf_nodes:
            if self.calculate_ucb(node) > self.calculate_ucb(node_to_expand):
                node_to_expand = node



    def calculate_ucb(self, node: MonteCarloNode, action) -> int:
        c = 1  # TODO: ?????
        N_s = node.number_of_visits
        N_s_a = node.edge_traversals.get(action)

        usa = c * np.sqrt(np.log(N_s)/(1 + N_s_a))
        return usa



    def get_action_from_actor(self, epsilon=0.05):
        # action = model.predict()
        # return action
        pass

    def train_model(self, actor_action, mcts_result):
        # model.train(actor_action, mcts_result)
        pass

    def calculate_error(self, actor_action, mcts_result):
        pass
