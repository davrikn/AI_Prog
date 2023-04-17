import sys

import configs
from hex.hexAgent import HexAgent
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
        get_ui = lambda model: HexUI(get_game(), model)
        # model = HexModel(configs.size, './model_dicts')
        model = HexModel(configs.size, snapshotdir=None)
    elif configs.game == 'nim':
        get_game = lambda: NimSimWorld(size=configs.size)
        get_ui = lambda model: NimUI(get_game(), model)
        model = NimModel(configs.size, './model_dicts')
    else:
        raise Exception(f"Game {configs.game} is not supported")

    total_episodes = 0
    simulations = 0
    while total_episodes <= configs.num_episodes:
        logger.debug(f"\nSimulation counter: {simulations + 1}")
        game = get_game()
        turns = 0
        utility = game.get_utility()
        while utility == 0:
            if total_episodes % 100 == 0:
                model.flush_rbuf()
            if total_episodes <= 200 and total_episodes % 100 == 0:
                model.save_model(file_name=f'hex_size_{model.size}_checkpoint_{total_episodes}')
                logger.info(f"Saved model at checkpoint: {total_episodes} episodes")
            next_game_state = MonteCarlo(root=game, model=model).run()
            # next_game_state = MonteCarlo(root=game).run()
            logger.debug(f"visited count of best edge: {next_game_state.visits}")
            turns += 1
            total_episodes += 1
            game = next_game_state.state

            if configs.display_UI:
                print(f'{game}\n')

            utility = next_game_state.state.get_utility()
        print(f"Player {1 if utility == 1 else 2} won")
        logger.debug(f"Player {1 if utility == 1 else 2} won")
        logger.debug(f"Total number of turns: {turns}")
    logger.info("Exiting")
    sys.exit(0)


if __name__ == "__main__":
    main()
