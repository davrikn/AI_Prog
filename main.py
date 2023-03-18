import sys

import configs
from nim.NimWorld import NimSimWorld
from nim.nimUI import NimUI
from nim.nimModel import NimModel
from hex.hexWorld import HexWorld
from hex.hexUI import HexUI
from hex.hexModel import HexModel
from monteCarlo import MonteCarlo
from typing import Callable
from model import Model
from game import Game
import logging
from tqdm import trange

logging.basicConfig(format="%(levelname)s %(asctime)s: %(message)s")

logger = logging.getLogger()
logger.setLevel(configs.log_level)

def main():
    get_game: Callable[[], Game] = lambda: None
    get_ui: Callable[[], HexUI or NimUI] = lambda: None
    model: Model = None
    if configs.game == 'hex':
        get_game = lambda: HexWorld(size=configs.size)
        get_ui = lambda: HexUI(get_game())
        model = HexModel(configs.size, './model_dicts')
    elif configs.game == 'nim':
        get_game = lambda: NimSimWorld(size=configs.size)
        get_ui = lambda: NimUI(get_game())
        model = NimModel(configs.size, './model_dicts')
    else:
        raise Exception(f"Game {configs.game} is not supported")

    if configs.ui:
        ui = get_ui()
        ui.start_game()
    else:
        for i in trange(configs.simulations):
            if i % 10 == 0:
                logger.info(f"On simulation {i}/{configs.simulations}")
            logger.debug(f"\nSimulation counter: {i + 1}")
            game = get_game()
            turns = 0
            while True:
                next_game_state = MonteCarlo(root=game, model=model).run()
                logger.debug(f"visited count of best edge: {next_game_state.visits}")
                turns += 1
                game = next_game_state.state

                utility = next_game_state.state.get_utility()
                if utility == 1:
                    logger.debug("player 1 won")
                    break
                elif utility == -1:
                    logger.debug("player 2 won")
                    break


    model.flush_rbuf()
    logger.info("Exiting")
    sys.exit(0)

if __name__ == "__main__":
    main()
