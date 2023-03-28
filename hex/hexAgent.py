import random

from hex.hexWorld import HexWorld
from hex.hexModel import HexModel


class HexAgent:
    def __init__(self, model: HexModel, name: str):
        self.model = model
        self.name = name
        self.wins = 0
        self.losses = 0

    def perform_move_probabilistic(self, state: HexWorld) -> HexWorld:
        actions_probabilities = self.model.classify(state.state(deNested=True))
        actions = [action[0] for action in actions_probabilities]
        weights = [action[1] for action in actions_probabilities]

        # actions.reverse()
        illegal_moves = 0
        found_legal_move = False
        while not found_legal_move:
            try:
                state.apply(random.choices(actions, weights=weights, k=1)[0])
                found_legal_move = True
            except:
                illegal_moves += 1
                pass
        return state

    def perform_move_greedy(self, state: HexWorld) -> HexWorld:
        actions = self.model.classify(state.state(deNested=True))

        # actions.reverse()
        illegal_moves = 0
        for action in actions:
            try:
                state.apply(action)
                break
            except:
                illegal_moves += 1
                pass
        # print(illegal_moves)
        return state


