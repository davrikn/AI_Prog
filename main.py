import configs
from MonteCarlo4 import MonteCarlo
from gameWorld import GameWorld
from hexWorld import HexWorld
from ui.hexgame_ui import HexGame
from nim.NimSimWorld import NimSimWorld
from nim.nimUI import NimUI


def main():
    # board_size = 8
    # hexWorld = HexWorld(board_size)
    # game = HexGame(hexWorld)
    # game.start_game()

    # game = NimSimWorld()

    # ui = NimUI(game)
    # ui.start_game()

    # states = game.get_child_states_enumerated()
    # for state in states:
    #     print(state)

    for i in range(configs.simulations):
        game = NimSimWorld(size=configs.size)
        mcts = MonteCarlo(game=game)
        mcts.perform_sim()




if __name__ == "__main__":
    main()
