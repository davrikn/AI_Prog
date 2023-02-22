import random
from functools import reduce
from game import Game
from typing import TypeVar, Callable
from math import log, sqrt
from configs import input_variables, num_episodes, decay_rate
from torch import nn
import torch


Node = TypeVar("Node", bound="MonteCarloNode")




class MonteCarloNode:
    depth = 0
    visits = 0
    total_score = 0
    children = []


    def __init__(self, state: Game, action: str, parent: Node or None, player=1):
        self.state = state
        self.parent = parent
        if self.parent is not None:
            self.depth = parent.depth + 1
        self.player = player
        self.action = action

    def expand(self) -> None:
        children = self.state.get_children_states()
        self.children = [MonteCarloNode(state=x[1], action=x[0], parent=self, player=self.player * -1) for x in children]

    def expanded(self) -> bool:
        return self.visits > 0

    def value(self) -> float:
        return self.total_score / self.visits

    def __str__(self, level=0) -> str:
        ret = "\t"*level, self.state.enumerate_state()
        for child in self.children:
            ret += child.__str__(level+1)

    def select_best_child(self) -> Node:
        ## TODO create func that uses at = Q(s,a) + N(s,a) and update key
        self.children.sort(reverse=True)
        return self.children[0]

    def __lt__(self, other: Node):
        return self.value() < other.value()

    def update_value(self, utility: float):
        self.visits += 1
        self.total_score += utility
        if self.parent is not None:
            self.parent.update_value(utility*decay_rate)


class MonteCarlo:

    def __init__(self, root=Game):
        self.root = MonteCarloNode(state=root)

    def run(self):
        for i in range(num_episodes):
            node = self.root

            while node.expanded():
                node = node.select_best_child()

            node.expand()
            node.update_value(self.rollout(node))


    def rollout(self, node: MonteCarloNode):
        if node.state.is_final_state():
            return node.state.get_state_utility()
        children = [x[1] for x in node.state.get_children_states()]
        return self.rollout(random.choice(children))