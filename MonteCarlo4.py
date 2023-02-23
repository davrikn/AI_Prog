import random
from functools import reduce
from game import Game
from typing import TypeVar, Callable
from math import log, sqrt
from configs import input_variables, num_episodes
from torch import nn
import torch

Node = TypeVar("Node", bound="MonteCarloNode")


class MonteCarloNode:
    depth = 0

    def __init__(self, state: Game, action: str or None, parent: Node or None, value_func: Callable[[], float]):
        self.state = state
        self.action = action
        self.parent = parent
        if self.parent is not None:
            self.depth = parent.depth + 1
        self.value = value_func


def node_value(x: MonteCarloNode):
    return x.value()


class N:
    dictionary = dict()

    def increment_state(self, state: str):
        if self.dictionary[state] is None:
            self.dictionary[state] = dict({"n": 1})
        else:
            self.dictionary[state].n += 1

    def increment_state_action(self, state: str, action: str):
        if self.dictionary[state][action] is None:
            self.dictionary[state][action] = 1
        else:
            self.dictionary[state][action] += 1

    def get_state(self, state: str):
        if self.dictionary.get(state) is None:
            return 0
        else:
            return self.dictionary[state].n

    def get_state_action(self, state: str, action: str):
        if self.dictionary[state][action] is None:
            return 0
        else:
            return self.dictionary[state][action]

    def get_best_action(self, state: str) -> str:
        if self.dictionary[state] is None:
            raise Exception("No data on state " + state)

        best_key = None
        best_key_value = 0

        for key in self.dictionary[state].keys():
            if key == "n":
                continue
            if self.dictionary[state][key] > best_key_value:
                best_key = key
                best_key_value = self.dictionary[state][key]
        return best_key


class Q:
    dictionary = dict()

    def update_state_action(self, state: str, action: str, value: float):
        if self.dictionary[state] is None:
            self.dictionary[state] = dict()

        if self.dictionary[state][action] is None:
            self.dictionary[state][action] = list([value])
        else:
            self.dictionary[state][action].append(value)

    def sum(self, a: float, b: float):
        return a + b

    def get_state_action(self, state: str, action: str):
        if self.dictionary[state] is None:
            self.dictionary[state] = dict()

        if self.dictionary[state][action] is None:
            return 0
        else:
            return reduce(self.sum, self.dictionary[state][action]) / len(self.dictionary[state][action])


class MonteCarlo:
    decay = 0.75
    c = 1

    N = N()
    Q = Q()

    def __init__(self, game: Game):
        self.initial_state = MonteCarloNode(state=game,
                                            parent=None,
                                            action=None,
                                            value_func=game.get_state_utility)

    def get_best_action(self, state: str):
        N.get_best_action(state)

    def perform_sim(self):
        for i in range(0, num_episodes):
            leaf = self.traverse2(self.initial_state)
            utility = self.rollout(leaf)
            self.backpropagate(leaf, float(utility))

    def backpropagate(self, node: MonteCarloNode, utility: float):
        self.Q.update_state_action(node.state.enumerate_state(), node.action, utility)
        if node.parent is not None:
            self.backpropagate(node.parent, utility * self.decay)

    def traverse2(self, current: MonteCarloNode) -> MonteCarloNode:
        if self.N.get_state(current.state.enumerate_state()) == 0:
            return current
        action_function: Callable[[float, float], float] = (lambda q, u: q + u) if current.depth % 2 == 0 else (
            lambda q, u: q - u)

        children = [MonteCarloNode(x[1], x[0], current, lambda: action_function(
            self.Q.get_state_action(current.state.enumerate_state(), x[0]), self.calc_usa(current, x[0]))) for x in
                    current.state.get_children_states()]
        children.sort(key=node_value)
        self.traverse2(children[0])

    def rollout(self, node: MonteCarloNode) -> int:
        print("test123", node.state.is_final_state())
        if node.state.is_final_state():
            return node.state.get_state_utility()
        children = [x[1] for x in node.state.get_children_states()]
        self.rollout(random.choice(children))

    def calc_usa(self, node: Node, action: str):
        n_s = log(self.N.get_state(node.state.enumerate_state()))
        n_s_a = self.N.get_state_action(node.state.enumerate_state(), action)
        return self.c * sqrt(n_s / (1 + n_s_a))


class ActorNetwork(nn.Module):
    input_length = input_variables

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
