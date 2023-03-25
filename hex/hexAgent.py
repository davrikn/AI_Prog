from hex.hexWorld import HexWorld
from hex.hexModel import HexModel


class HexAgent:
    def __init__(self, model: HexModel, name: str):
        self.model = model
        self.name = name
        self.wins = 0
        self.losses = 0

    def perform_move(self, state: HexWorld) -> HexWorld:
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


