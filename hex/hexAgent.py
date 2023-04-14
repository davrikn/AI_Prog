import math
import random
import numpy as np
import torch

import configs
from hex.hexWorld import HexWorld
from hex.hexModel import HexModel


class HexAgent:
    def __init__(self, model: HexModel, name: str):
        self.model = model
        self.name = name
        self.wins = 0
        self.losses = 0

    def perform_move_probabilistic(self, state: HexWorld) -> HexWorld:
        def remove_invalid_moves(action_dist, state: HexWorld):
            valid_actions = state.get_possible_actions()
            for idx, action in enumerate(action_dist):
                if action[0] not in valid_actions:
                    action_dist.remove(action)

        actions_probabilities = self.model.classify(state.state(deNested=True))
        remove_invalid_moves(actions_probabilities, state)
        actions = [action[0] for action in actions_probabilities]
        weights = [action[1]**(1/4) for action in actions_probabilities]
        print("pre", weights)
        weights = self.model.sm(torch.tensor(weights, dtype=torch.float)).numpy()
        print("post", weights)
        print("------------------------------------------------------------")
        # min_val = min(weights)
        # if min_val < 0:
        #     weights = [(x + abs(min_val) + 1)**3 for x in weights]


        # actions.reverse()
        illegal_moves = 0
        found_legal_move = False
        while not found_legal_move:
            try:
                if illegal_moves >= len(actions):
                    action = random.choices(list(self.model.action_to_index.keys()))[0]
                    state.apply(action)
                    found_legal_move = True
                state.apply(random.choices(actions, weights=weights, k=1)[0])
                found_legal_move = True
            except:
                illegal_moves += 1
                pass

        return state

    def perform_move_greedy(self, state: HexWorld) -> HexWorld:
        action_probabilities = self.model.classify(state.state(deNested=True))
        actions = [action[0] for action in action_probabilities]
        # if state.player == -1:
        #     actions.reverse()

        # actions.reverse()
        illegal_moves = 0
        for action in actions:
            try:
                state.apply(action)
                break
            except:
                illegal_moves += 1
                pass
        if illegal_moves == len(action_probabilities):
            raise Exception
        return state

    def perform_move_random(self, state: HexWorld) -> HexWorld:
        action_probabilities = self.model.classify(state.state(deNested=True))
        actions = [action[0] for action in action_probabilities]
        random.shuffle(actions)

        # actions.reverse()
        illegal_moves = 0
        for action in actions:
            try:
                state.apply(action)
                break
            except:
                illegal_moves += 1
                pass
        if illegal_moves == len(action_probabilities):
            raise Exception
        return state

    def get_action(self, state: list[int], player_to_move: int) -> tuple[int, int]:
        player_id = state.pop(0)
        player1_board = np.array([0 if x == 2 else x for x in state])
        player2_board = np.array([0 if x == 1 else x for x in state])
        player2_board = np.array([1 if x == 2 else x for x in player2_board])

        player1_board = np.reshape(player1_board, (configs.OHT_size, configs.OHT_size))
        player2_board = np.reshape(player2_board, (configs.OHT_size, configs.OHT_size))
        board = np.stack((player1_board, player2_board), axis=0)

        player_to_move = 1 if player_to_move == 1 else -1
        x = (board, player_to_move)

        pred = self.model.classify(x)
        actions = [action[0] for action in pred]
        actions_tuples = []
        for action in actions:
            row = list(action)[1]
            column = list(action)[-1]
            actions_tuples.append((int(row), int(column)))

        full_board = np.reshape(state, (configs.OHT_size, configs.OHT_size))
        for move in actions_tuples:
            if full_board[move[0]][move[1]] == 0:
                return move





