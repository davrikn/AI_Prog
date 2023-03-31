import random

import configs
from hex.hexAgent import HexAgent
from hex.hexModel import HexModel
from hex.hexWorld import HexWorld
from monteCarlo import MonteCarlo
from tournament import Tournament

# agent0 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_5_checkpoint_0.pt")
#                   , name="Checkpoint0Agent")
# agent50 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_5_checkpoint_50.pt")
#                    , name="Checkpoint50Agent")
# agent100 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_5_checkpoint_100.pt")
#                     , name="Checkpoint100Agent")
# agent150 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_5_checkpoint_150.pt")
#                     , name="Checkpoint150Agent")
# agent200 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_5_checkpoint_200.pt")
#                     , name="Checkpoint200Agent")

test0 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/0.pt"), name="0")
test50 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/50.pt"), name="50")
test100 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/100.pt"), name="100")
test150 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/150.pt"), name="150")
test200 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/200.pt"), name="200")
test300 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/300.pt"), name="300")

# tournament = Tournament([agent0, agent50, agent100, agent150, agent200], UI=False)
tournament = Tournament([test0, test50, test100, test150, test200, test300], G=250, UI=False)


tournament.run_tournament()

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


# agent_vs_mcts()
