import configs
from nim.NimWorld import NimSimWorld
from nim.nimUI import NimUI
from hex.hexWorld import HexWorld
from hex.hexUI import HexUI
from monteCarlo import MonteCarlo
from gameWorld import GameWorld
from typing import Callable

def main():
    get_game: Callable[[], GameWorld] = lambda : None
    get_ui: Callable[[], HexUI or NimUI] = lambda : None
    if configs.game == 'hex':
        get_game = lambda : HexWorld(size=configs.size)
        ui = HexUI(get_game())
    elif configs.game == 'nim':
        get_game = lambda : NimSimWorld(size=configs.size)
        ui = NimUI(get_game())
    else:
        raise Exception(f"Game {configs.game} is not supported")

    for i in range(configs.simulations):
        print("\nSimulation counter:", i + 1)
        game = get_game()
        turns = 0
        curr_player = 1
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
