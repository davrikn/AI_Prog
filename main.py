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

    # states = game.get_child_states_enumerated()
    # for state in states:
    #     print(state)

    def run_with_ui():
        game = NimSimWorld(size=configs.size)
        ui = NimUI(game)
        ui.start_game()

    # run_with_ui()


    for i in range(configs.simulations):
        print("\nSimulation counter:", i + 1)
        game = NimSimWorld(size=configs.size)
        turns = 0
        curr_player = -1
        while True:
            print("\nplayers turn: ", curr_player)
            next_game_state = MonteCarlo(root=game, player=curr_player).run()
            print("visited count of best edge: ", next_game_state.visits)
            turns += 1
            game = next_game_state.state
            curr_player = next_game_state.player

            if next_game_state.state.is_final_state():
                print("The game ended after", turns, "turns")
                if next_game_state.player == 1:
                    print("player 1 won")
                else:
                    print("player 2 won")
                break


if __name__ == "__main__":
    main()
