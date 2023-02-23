import random
from functools import reduce

import numpy as np

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
        self.children = [MonteCarloNode(state=x[1], action=x[0], parent=self, player=self.player * -1) for x in
                         children]

    def expanded(self) -> bool:
        return self.visits > 0

    def value(self) -> float:
        # return self.total_score / self.visits + sqrt(log(self.parent.visits) / (1 + self.visits))
        return sqrt(log(self.parent.visits) / (1 + self.visits))

    def __str__(self, level=0) -> str:
        ret = "\t" * level + self.state.enumerate_state()
        ret += " visits: " + str(self.visits) + " Q: " + str(self.total_score / self.visits) + "\n"
        for child in self.children:
            ret += child.__str__(level + 1)

    def select_best_child(self) -> Node:
        ## TODO create func that uses at = Q(s,a) + N(s,a) and update key
        self.children.sort(reverse=True)
        return self.children[0]

    ## Fix random bug
    def select_best_child2(self) -> Node:
        if self.player == 1:
            self.children.sort(reverse=True, key=self.a_t_max)
            return self.children[0]
            # for child in self.children:
            #     if child.get_q_s_a() + child.get_u_s_a() > a_t.get_q_s_a() + a_t.get_u_s_a():
            #         a_t = child
        if self.player == -1:
            print(self.children)
            self.children.sort(key=self.a_t_min)
            return self.children[0]
            # for child in self.children:
            #     if child.get_q_s_a() - child.get_u_s_a() < a_t.get_q_s_a() - a_t.get_u_s_a():
            #         a_t = child

    def a_t_max(self, node):
        return node.get_q_s_a() + node.get_u_s_a()

    def a_t_min(self, node):
        return self.get_q_s_a() - self.get_u_s_a()

    def get_q_s_a(self):
        if self.visits == 0:
            return np.inf
        return self.total_score / self.visits

    def get_u_s_a(self):
        return sqrt(log(self.parent.visits) / (1 + self.visits))

    def __lt__(self, other: Node):
        return self.value() < other.value()

    def update_value(self, utility: float):
        self.visits += 1
        self.total_score += utility
        if self.parent is not None:
            self.parent.update_value(utility * decay_rate)


class MonteCarlo:

    def __init__(self, root=Game):
        self.root = MonteCarloNode(state=root, action="", parent=None)

    def run(self) -> Game:
        for i in range(num_episodes):
            node = self.root

            while node.expanded():
                node = node.select_best_child()

            node.expand()
            node.update_value(self.rollout(node))
        return self.select_best_edge()

    def select_best_edge(self) -> Game:
        self.root.children.sort(reverse=True, key=lambda child: child.visits)
        return self.root.children[0].state

    def rollout(self, node: MonteCarloNode):
        if node.state.is_final_state():
            return node.state.get_state_utility()
        children = [MonteCarloNode(state=x[1], parent=node, action=x[0], player=node.player * -1)
                    for x in node.state.get_children_states()]
        return self.rollout(random.choice(children))
