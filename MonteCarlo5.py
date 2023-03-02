import csv
import random
from functools import reduce, cmp_to_key
from os.path import exists

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

    def __init__(self, state: Game, action: str, parent: Node or None, player: int):
        self.state = state
        self.parent = parent
        if self.parent is not None:
            self.depth = parent.depth + 1
        self.player = player
        self.action = action

    def expand(self) -> None:
        children = self.state.get_children_states()
        if len(children) == 0:
            raise Exception("Trying to expand terminal state")
        self.children = [MonteCarloNode(state=x[1], action=x[0], parent=self, player=self.player * -1) for x in
                         children]

    def is_expanded(self) -> bool:
        return self.visits > 0

    def value(self) -> float:
        # return self.total_score / self.visits + sqrt(log(self.parent.visits) / (1 + self.visits))
        return sqrt(log(self.parent.visits) / (1 + self.visits))

    def __str__(self, level=0) -> str:
        ret = "\t" * level + self.state.enumerate_state()
        ret += " visits: " + str(self.visits) + " Q: " + str(self.total_score / self.visits) + "\n"
        for child in self.children:
            ret += child.__str__(level + 1)

    def select_next_child(self) -> Node:
        if self.player == 1:
            return sorted(self.children, reverse=True,
                          key=cmp_to_key(lambda node1, node2: node1.a_t_max() - node2.a_t_max()))[0]
        if self.player == -1:
            return sorted(self.children,
                          key=cmp_to_key(lambda node1, node2: node1.a_t_min() - node2.a_t_min()))[0]

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

    def __lt__(self, other: Node):
        return self.value() < other.value()

    # def update_value(self, utility: float):
    #     self.visits += 1
    #     self.total_score += utility
    #     if self.parent is not None:
    #         # self.parent.update_value(utility * decay_rate)  # TODO: Discuss decay rate
    #         self.parent.update_value(utility)


class MonteCarlo:

    def __init__(self, player, root=Game):
        self.root = MonteCarloNode(state=root, action="", parent=None, player=player)

    def run(self) -> MonteCarloNode:
        for i in range(num_episodes):
            # Tree Search using Tree Policy
            leaf_node = self.tree_search()

            # Stop the loop if the entire state-tree is generated
            if leaf_node.state.is_final_state():
                break

            # Expand the leaf node
            leaf_node.expand()

            # Leaf evaluation using rollout simulation
            utility = self.rollout(leaf_node)

            # Backpropagation - Passing the utility of the final state back up the tree
            self.backpropagation(leaf_node, utility)

        self.create_train_data()

        return self.get_most_visited_edge()

    def get_most_visited_edge(self) -> MonteCarloNode:
        print("total children visits: ", self.get_total_children_visits())
        return sorted(self.root.children, reverse=True, key=lambda child: child.visits)[0]

    # TODO: Delete this (debugging method)
    def get_total_children_visits(self):
        tot = 0
        for child in self.root.children:
            tot += child.visits
        return tot

    def tree_search(self):
        """
        Uses the tree-policy to traverse the tree until a leaf node is found
        :return: Leaf node
        """

        node = self.root

        while node.is_expanded():
            node = node.select_next_child()
        return node

    def rollout(self, node: MonteCarloNode):
        if node.state.is_final_state():
            # returns negative score for second player since player 2 is <-1>
            return node.state.get_state_utility() * node.player

        children = [MonteCarloNode(state=x[1], parent=node, action=x[0], player=node.player * -1)
                    for x in node.state.get_children_states()]
        return self.rollout(random.choice(children))

    def backpropagation(self, node: MonteCarloNode, value: int):
        node.visits += 1
        node.total_score += value

        if node.parent is not None:
            self.backpropagation(node.parent, value)

    def get_action_distribution(self, node: MonteCarloNode):
        visits = [x.visits for x in node.children]
        if 0 in visits:
            return None

        visit_sum = np.sum(visits)
        dists = [(x.action, x.visits / visit_sum) for x in node.children]

        return dists

    def create_train_data(self):
        dists = self.get_action_distribution(self.root)
        if dists is None:
            return

        if not exists('train.csv'):
            with open('train.csv', 'w', newline='') as f:
                writer = csv.writer(f)

                fields = ["state", "player", "mcts_distribution"]
                writer.writerow(fields)

                f.close()

        with open('train.csv', 'a', newline='') as f:
            writer = csv.writer(f)

            state = self.root.state.enumerate_state2()
            player = self.root.player
            dists = self.get_action_distribution(self.root)
            print(dists)

            writer.writerow([state, player, dists])

        f.close()
