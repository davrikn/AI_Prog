import random

import configs
from configs import size
from typing import TypeVar
from game import Game
import numpy as np

Node = TypeVar("Node", bound="MonteCarloNode")


class MonteCarloNode:
    def __init__(self, state: Game, parent: Node or None):
        self.parent = parent
        self.children: list[MonteCarloNode] = []
        self.state = state
        self.n = 0
        self.total_rewards = 0
        self.traversals_of_actions = self.init_traversals_of_action_dict()

    def init_traversals_of_action_dict(self) -> dict:
        action_dict = dict()
        actions: list[str] = self.state.get_child_states_enumerated()

        for action in actions:
            action_dict[action] = 0
        return action_dict

    def increment_n(self):
        self.n += 1

    def increment_total_reward(self, reward):
        self.total_rewards += reward

    def increment_action_traversal(self, action: str):
        self.traversals_of_actions[action] += 1

    def get_q_value(self):
        return self.total_rewards / self.n


class MonteCarlo:
    def __init__(self, initial_state=Game(size)):
        self.root_node = MonteCarloNode(initial_state, None)
        self.current_node = self.root_node

    def perform_simulation(self):
        self.generate_child_nodes(self.current_node)

        for i in range(configs.num_episodes):
            self.current_node = self.select_next_node_ucb()
            if self.current_node.state.is_final_state():
                print("final state")
                break
            if self.current_node.n == 0:
                leaf_node = self.current_node
                reward = self.rollout(leaf_node)
                self.backpropagation(reward, leaf_node)

    def traverse_node(self, node: Node, action: str):
        if action not in node.state.get_child_states_enumerated():
            raise Exception("Illegal action")

        child_node = MonteCarloNode(state=Game(action), parent=node)
        node.children.append(child_node)
        return child_node

    def select_next_node_ucb(self) -> Node:
        """Return a Node

        Uses the Upper Confidence Bounds (UCB) to select which leaf node to expand
        """
        # child_nodes = self.generate_child_nodes(self.current_node)
        node_to_expand = self.current_node.children[0]
        best_ucb = self.calculate_usa(self.current_node, node_to_expand.state.enumerate_state())
        for node in self.current_node.children:
            node_ucb = node.get_q_value() + self.calculate_usa(self.current_node, node.state.enumerate_state())
            if node_ucb > best_ucb:
                node_to_expand = node
                best_ucb = node_ucb
        return node_to_expand

    def calculate_usa(self, node: Node, action: str) -> int:
        n_s = node.n
        n_s_a = node.traversals_of_actions.get(action)
        u_s_a = np.sqrt(np.log(n_s) / (1 + n_s_a))
        return u_s_a

    def rollout(self, node: Node):
        reward = 0
        while True:
            if node.state.is_final_state():
                print("final state")
                reward = 0  # TODO: rewards
                break

            actions = node.state.get_child_states_enumerated()
            random_action = random.choice(actions)
            node = self.traverse_node(node, random_action)
        return 0

    def backpropagation(self, reward, node: Node):
        while node.parent is not None:
            node.n += 1
            node.total_rewards += reward
            # TODO: increment actions traversed
            node = node.parent

    def get_action_from_actor(self, epsilon=0.05):
        # action = model.predict()
        # return action
        pass

    def train_model(self, actor_action, mcts_result):
        # model.train(actor_action, mcts_result)
        pass

    def calculate_error(self, actor_action, mcts_result):
        pass

    def generate_child_nodes(self, node: Node):
        child_states = node.state.get_children_states()
        node.children = [MonteCarloNode(state=child, parent=node) for child in child_states]
