from gameWorld import GameWorld
from hexWorld import HexWorld
from ui.hexgame_ui import HexGame


def main():
    board_size = 8
    game = HexGame(board_size)
    game.start_game()

    # print(hex_game.world)


if __name__ == "__main__":
    main()
