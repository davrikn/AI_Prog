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

logging.basicConfig(format="%(levelname)s %(asctime)s: %(message)s")

logger = logging.getLogger()
logger.setLevel(configs.log_level)

class ReinforcementLearning:

    def run(self):
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

        saving_intervals = (configs.num_episodes / (configs.M - 1))
        if not saving_intervals.is_integer():
            raise Exception(f"Num episodes {configs.num_episodes} must be divisible by M-1 {configs.M - 1}")

        episodes = 0
        while episodes <= configs.num_episodes:
            print(f"\nEpisode counter: {episodes + 1}")
            game = get_game()
            turns = 0
            utility = game.get_utility()
            while utility == 0:
                next_game_state = MonteCarlo(root=game, model=model).run()
                logger.debug(f"visited count of best edge: {next_game_state.visits}")
                turns += 1
                game = next_game_state.state
                if configs.display_UI:
                    print(f'{game}\n')
                utility = next_game_state.state.get_utility()

            print(f"Player {1 if utility == 1 else 2} won")
            logger.debug(f"Player {1 if utility == 1 else 2} won")
            logger.debug(f"Total number of turns: {turns}")

            if episodes % saving_intervals == 0:
                model.save_model(file_name=f'{configs.game}_size_{model.size}_checkpoint_{episodes}')
                logger.info(f"Saved model at checkpoint: {episodes} episodes")
            model.flush_rbuf()
            episodes += 1
            configs.epsilon *= configs.epsilon_decay
            if configs.epsilon < 0.2:
                configs.epsilon = 0.2
        logger.info("Exiting")
        sys.exit(0)


if __name__ == "__main__":
    rf = ReinforcementLearning()
    rf.run()
