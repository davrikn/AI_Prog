import configs
from MonteCarlo5 import MonteCarlo
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
    game = NimSimWorld(size=configs.size, player=1)
    for i in range(configs.simulations):
        print("Move counter: ", i)
        game = MonteCarlo(root=game).run()




if __name__ == "__main__":
    main()
