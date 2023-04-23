import csv
import logging
import math
import random
import time
from functools import reduce, cmp_to_key
from os.path import exists
from hex.hexWorld import HexWorld
from hex.hexModel import HexModel

import numpy as np

import configs
from game import Game
from typing import TypeVar, Callable
from math import log, sqrt
from configs import input_variables, num_rollouts, decay_rate
from model import Model
from torch import nn
import torch

Node = TypeVar("Node", bound="MonteCarloNode")


logger = logging.getLogger()

class MonteCarloNode:
    visits = 0
    total_score = 0
    children: list[Node] = []

    def __init__(self, state: Game, action: str = "", parent: Node = None):
        self.state = state
        self.action = action
        self.parent = parent

    def expand(self) -> None:
        children = self.state.get_children_states()
        if len(children) == 0:
            raise Exception("Trying to expand terminal state")
        self.children = [MonteCarloNode(state=child[1], action=child[0], parent=self) for child in children]

    def is_expanded(self) -> bool:
        return self.visits > 0

    def a_t_max(self):
        return self.get_q_s_a() + self.get_u_s_a()

    def a_t_min(self):
        return self.get_q_s_a() - self.get_u_s_a()

    def get_q_s_a(self):
        if self.visits == 0:
            return 0
        return self.total_score / self.visits

    def get_u_s_a(self):
        if self.visits == 0:
            return np.inf
        return sqrt(log(self.parent.visits) / self.visits)

    def select_next_child(self) -> Node:
        if self.state.player == 1:
            return sorted(self.children, reverse=True,
                          key=cmp_to_key(lambda node1, node2: node1.a_t_max() - node2.a_t_max()))[0]
        if self.state.player == -1:
            return sorted(self.children,
                          key=cmp_to_key(lambda node1, node2: node1.a_t_min() - node2.a_t_min()))[0]

    def __str__(self, level=0) -> str:
        ret = "\t" * level + self.state.state_stringified()
        ret += " visits: " + str(self.visits) + " Q: " + str(0 if self.visits == 0 else self.total_score / self.visits) + "\n"
        for child in sorted(self.children, key=cmp_to_key(lambda x,y: x.visits - y.visits), reverse=True):
            ret += child.__str__(level + 1)
        return ret

class MonteCarlo:
    rollouts = 0
    rollout_time = 0
    expands = 0
    expand_time = 0

    def __init__(self, root: Game, model: Model = None):
        self.model = model
        self.root = MonteCarloNode(state=root)

    def run(self) -> MonteCarloNode:
        for i in range(num_rollouts):
            # Tree Search using Tree Policy
            leaf_node = self.tree_search()

            # Expand the leaf node
            expand_start = time.perf_counter()
            if not leaf_node.state.is_final_state():
                leaf_node.expand()

            # self.check_child_is_win(leaf_node.children)

            expand_end = time.perf_counter()
            self.expands += 1
            self.expand_time += expand_end-expand_start

            # Leaf evaluation using rollout simulation
            rollout_start = time.perf_counter()
            utility = self.rollout(leaf_node)
            rollout_end = time.perf_counter()
            self.rollouts += 1
            self.rollout_time += rollout_end-rollout_start

            # Backpropagation - Passing the utility of the final state back up the tree
            self.backpropagate(leaf_node, utility)
            if i == num_rollouts-1:
                logger.debug(f"Exiting after {num_rollouts} episodes")
        logger.debug(f"Average rollout duration: {self.rollout_time/self.rollouts}. Rollout count: {self.rollouts}. Total rollout time: {self.rollout_time}")
        logger.debug(f"Average expand duration: {self.expand_time/self.expands}. Expand count: {self.expands}. Total expand time: {self.expand_time}")

        self.flush_train_data()
        return self.get_most_visited_edge()

    def get_most_visited_edge(self) -> MonteCarloNode:
        highest_visit_count = sorted(self.root.children, reverse=True, key=lambda child: child.visits)[0].visits
        most_visited_edges = [self.root.children[i] for i in range(len(self.root.children))
                              if self.root.children[i].visits == highest_visit_count]

        return random.choice(most_visited_edges)
        # return sorted(self.root.children, reverse=True, key=lambda child: child.visits)[0]

    def tree_search(self):
        """
        Uses the tree-policy to traverse the tree until a leaf node is found
        :return: Leaf node
        """
        node = self.root

        while node.is_expanded():
            if node.state.is_final_state():
                return node
            node = node.select_next_child()
        return node

    def rollout(self, node: MonteCarloNode):
        if node.state.is_final_state():
            return node.state.get_utility()

        if self.model is None or random.random() < configs.epsilon:
            children = [MonteCarloNode(state=x[1], action=x[0], parent=node) for x in node.state.get_children_states()]
            return self.rollout(random.choice(children))
        else:
            actions = self.model.classify(node.state.state(deNested=True))
            actions = [action[0] for action in actions]
            for action in actions:
                try:
                    child = node.state.apply(action, deepcopy=True)
                    return self.rollout(MonteCarloNode(state=child, parent=node, action=action))
                except:
                    pass
        raise Exception("No action found")

    def backpropagate(self, node: MonteCarloNode, value: float):
        node.visits += 1
        node.total_score += value

        if node.parent is not None:
            self.backpropagate(node.parent, value * configs.Sdecay_rate)

    def flush_train_data(self):
        if self.model is None:
            return
        self.model.append_rbuf_single((self.root.state.state(deNested=True), [(x.action, x.visits / (self.root.visits-1)) for x in self.root.children]))


if __name__ == "__main__":
    logging.basicConfig()
    logger.setLevel(configs.log_level)
    mc = MonteCarlo(root=HexWorld(configs.size), model=None)
    mc.run()
    #print(mc.root)
