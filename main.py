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

    game = NimSimWorld()
    # game.produce_initial_state()
    # game.print_board()
    # states = game.get_children_states()
    # print(len(states))
    # game.pick_piece_from_pile(0)
    # game.pick_piece_from_pile(1)
    # game.pick_piece_from_pile(1)
    # game.pick_piece_from_pile(2)
    # game.pick_piece_from_pile(2)
    # game.pick_piece_from_pile(2)
    # print(game.is_final_state())

    # ui = NimUI(game)
    # ui.start_game()

    states = game.get_child_states_enumerated()
    for state in states:
        print(state)


if __name__ == "__main__":
    main()
