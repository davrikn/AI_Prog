from configs import size
from typing import TypeVar
from game import Game
import numpy as np

Node = TypeVar("Node", bound="MonteCarloNode")

class BookKeeper:
    data = dict()

    def increment_state(self, state: str) -> None:
        if self.data[state] is None:
            self.data[state] = dict({"n": 1})
        else:
            self.data[state].n += 1

    def increment_state_action(self, state: str, action: str) -> None:
        if self.data[state] is None:
            self.data[state] = dict({"n": 1})
        if self.data[state][action] is None:
            self.data[state][action] = 1
        else:
            self.data[state][action] += 1

    def get_state(self, state: str) -> int:
        if self.data[state] is None:
            return 0
        else:
            return self.data[state].n

    def get_state_action(self, state: str, action: str) -> int:
        if self.data[state] is None or self.data[state][action] is None:
            return 0
        else:
            return self.data[state][action]

class MonteCarloNode:
    def __init__(self, game: Game, parent: Node):
        self.parent = parent
        self.state = game


class MonteCarlo:
    def __init__(self, initial_state=Game(size)):
        self.initial_state = MonteCarloNode(initial_state)
        self.bookkeper = BookKeeper()

    def select_node_to_expand(self, leaf_nodes: list[MonteCarloNode]) -> MonteCarloNode:
        """Return a Node

        Uses the Upper Confidence Bounds (UCB) to select which leaf node to expand
        """
        # TODO: fant ikke en oneliner for dette, HJELP DAVID
        node_to_expand = leaf_nodes[0]
        for node in leaf_nodes:
            if self.calculate_ucb(node) > self.calculate_ucb(node_to_expand):
                node_to_expand = node



    def calculate_ucb(self, state: str, action: str) -> int:
        c = 1
        n_s = self.bookkeper.get_state(state)
        n_s_a = self.bookkeper.get_state_action(state, action)
        usa = c * np.sqrt(np.log(n_s)/(1 + n_s_a))
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
