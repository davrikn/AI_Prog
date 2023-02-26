import configs
from MonteCarlo5 import MonteCarlo
from game import Game
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
        print("Simulation counter:", i)
        game = NimSimWorld(size=configs.size)
        turns = 0
        player = 1
        while True:
            next_game_state = MonteCarlo(root=game, player=player).run()
            turns += 1
            game = next_game_state.state
            player = next_game_state.player
            if next_game_state.state.is_final_state():
                print("\nThe game ended after", turns, "turns")
                if next_game_state.player == 1:
                    print("player 1 won")
                else:
                    print("player 2 won")
                break


if __name__ == "__main__":
    main()
