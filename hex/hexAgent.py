import copy
import random
import numpy as np

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
            test = 0
            new_dist = []
            for idx, action in enumerate(action_dist):
                if action[0] in valid_actions:
                    new_dist.append(action)
            return new_dist

        def scale_weights(w):
            scaled_weights = [x**5 for x in w]
            return scaled_weights/np.sum(scaled_weights)


        actions_probabilities = self.model.classify(state.state(deNested=True))
        actions_probabilities = remove_invalid_moves(actions_probabilities, state)
        actions = [action[0] for action in actions_probabilities]
        weights = [action[1] for action in actions_probabilities]
        weights = scale_weights(weights)

        illegal_moves = 0
        found_legal_move = False
        while not found_legal_move:
            try:
                state.apply(random.choices(actions, weights=weights, k=1)[0])
                found_legal_move = True
            except:
                illegal_moves += 1
                # pass
                raise Exception("Invalid move..")

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
                print(action)
                print("Illegal moves: ", illegal_moves)
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

    def get_action(self, state: list[int], starting_player: int) -> tuple[int, int]:
        player_id = state.pop(0)
        state = np.reshape(state, (configs.OHT_size, configs.OHT_size))
        # state = np.rot90(state, k=1)
        board_cpy = copy.deepcopy(state)
        state = state.flatten()

        player1_board = np.array([0 if x == 2 else x for x in state])
        player2_board = np.array([0 if x == 1 else x for x in state])
        player2_board = np.array([1 if x == 2 else x for x in player2_board])

        player1_board = np.reshape(player1_board, (configs.OHT_size, configs.OHT_size))
        player2_board = np.reshape(player2_board, (configs.OHT_size, configs.OHT_size))

        board = np.stack((player1_board, player2_board), axis=0)

        if player_id == 1:
            x = (board, 1)
        else:
            x = (board, -1)
        # print("\n\n")
        # print(board_cpy)
        # print(x)

        pred = self.model.classify(x)
        actions = [action[0] for action in pred]

        actions = np.reshape(actions, (configs.OHT_size, configs.OHT_size))
        # actions = np.rot90(actions, k=-1)

        actions = actions.flatten()
        actions_tuples = []
        for action in actions:
            row = list(action)[1]
            column = list(action)[-1]
            actions_tuples.append((int(row), int(column)))
            # actions_tuples.append((int(column), int(row)))

        full_board = np.reshape(state, (configs.OHT_size, configs.OHT_size))
        # full_board = np.rot90(full_board, k=-1)
        illegal_moves = 0
        for move in actions_tuples:
            if full_board[move[0]][move[1]] == 0:
                print("Illegal moves: ", illegal_moves)
                return move
            else:
                illegal_moves += 1





