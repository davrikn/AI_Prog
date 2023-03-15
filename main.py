import configs
from nim.NimWorld import NimSimWorld
from nim.nimUI import NimUI
from nim.nimModel import NimModel
from hex.hexWorld import HexWorld
from hex.hexUI import HexUI
from hex.hexModel import HexModel
from monteCarlo import MonteCarlo
from gameWorld import GameWorld
from typing import Callable
from model import Model

def main():
    get_game: Callable[[], GameWorld] = lambda: None
    get_ui: Callable[[], HexUI or NimUI] = lambda: None
    model: Model = None
    if configs.game == 'hex':
        get_game = lambda: HexWorld(size=configs.size)
        get_ui = lambda: HexUI(get_game())
        model = HexModel(gamesize=configs.size)
    elif configs.game == 'nim':
        get_game = lambda: NimSimWorld(size=configs.size)
        get_ui = lambda: NimUI(get_game())
        model = NimModel(gamesize=configs.size)
    else:
        raise Exception(f"Game {configs.game} is not supported")

    if configs.ui:
        ui = get_ui()
        ui.start_game()
    else:
        for i in range(configs.simulations):
            print("\nSimulation counter:", i + 1)
            game = get_game()
            turns = 0
            curr_player = 1
            while True:
                print("\nplayers turn: ", curr_player)
                next_game_state = MonteCarlo(root=game, player=curr_player, model=model).run()
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
