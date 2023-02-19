from functools import reduce
from game import Game
from typing import TypeVar, Callable
from math import log, sqrt
from configs import input_variables
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
        if self.dictionary[state] is None:
            return 0
        else:
            return self.dictionary[state].n

    def get_state_action(self, state: str, action: str):
        if self.dictionary[state][action] is None:
            return 0
        else:
            return self.dictionary[state][action]


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
        return a+b

    def get_state_action(self, state: str, action: str):
        if self.dictionary[state] is None:
            self.dictionary[state] = dict()

        if self.dictionary[state][action] is None:
            return 0
        else:
            return reduce(self.sum, self.dictionary[state][action])/len(self.dictionary[state][action])


class MonteCarlo:
    c = 1

    N = N()
    Q = Q()
    def __init__(self, game: Game):
        self.initial_state = MonteCarloNode(game.produce_initial_state())

    # TODO: Re-do as recursive
    def traverse(self):
        frontier: list[MonteCarloNode] = list([self.initial_state])

        # TODO: Update in loop so that the best fit will be returned in case all nodes have been rolled out once
        best_fit = frontier[0]

        while len(frontier) != 0:
            node = frontier.pop()
            if self.N.get_state(node.state.enumerate_state()) is 0:
                return node

            action_function: Callable[[float, float], float] = (lambda q, u: q + u) if node.depth % 2 == 0 else (lambda q, u: q - u)

            children = [MonteCarloNode(x[1], x[0], node, lambda: action_function(self.Q.get_state_action(node.state.enumerate_state(), x[0]), self.calc_usa(node, x[0]))) for x in node.state.get_children_states()]
            frontier.extend(children)
            frontier.sort(key=node_value)
        return best_fit

    def rollout(self, node: MonteCarloNode) -> int:
        pass

    def calc_usa(self, node: Node, action: str):
        n_s = log(self.N.get_state(node.state.enumerate_state()))
        n_s_a = self.N.get_state_action(node.state.enumerate_state(), action)
        return self.c*sqrt(n_s / (1 + n_s_a))

class ActorNetwork(nn.Module):
    input_length = input_variables

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits