import random

import configs
from hex.hexAgent import HexAgent
from hex.hexModel import HexModel
from hex.hexWorld import HexWorld
from monteCarlo import MonteCarlo
from tournament import Tournament

agent0 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_5_checkpoint_0.pt")
                  , name="Checkpoint0Agent")
agent50 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_5_checkpoint_50.pt")
                   , name="Checkpoint50Agent")
agent100 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_5_checkpoint_100.pt")
                    , name="Checkpoint100Agent")
agent150 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_5_checkpoint_150.pt")
                    , name="Checkpoint150Agent")
agent200 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_5_checkpoint_200.pt")
                    , name="Checkpoint200Agent")

tournament = Tournament([agent0, agent50, agent100, agent150, agent200], UI=False)


# tournament.run_tournament()

# TODO: Delete, just testing
def agent_vs_mcts():
    for i in range(100):
        player_to_move = 0
        game = HexWorld(size=configs.size)
        while True:
            if player_to_move == 0:
                game = MonteCarlo(game, model=None).run()
                if game.state.is_final_state():
                    agent0.losses += 1
                    print("MCTS WON")
                    break
                player_to_move = 1
            else:
                game = agent0.perform_move_probabilistic(game.state)
                if game.is_final_state():
                    agent0.wins += 1
                    print("AGENT WON")
                    break
                player_to_move = 0
    print(f'agent wins: {agent0.wins}, agent losses: {agent0.losses}')


agent_vs_mcts()
