import random

import configs
from hex.hexAgent import HexAgent
from hex.hexModel import HexModel
from hex.hexWorld import HexWorld
from monteCarlo import MonteCarlo, MonteCarloNode
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

test0 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/0.pt"), name="__________0")
test100 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/100.pt"), name="__________100")
test200 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/200.pt"), name="__________200")
test300 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/300.pt"), name="__________300")
test400 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/400.pt"), name="__________400")
test600 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/600.pt"), name="__________600")
test800 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/800.pt"), name="__________800")
test1000 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/1000.pt"), name="__________1000")

# tournament = Tournament([agent0, agent50, agent100, agent150, agent200], UI=False)
tournament = Tournament([test200, test300, test400, test0, test1000, test100], G=150, UI=True)


tournament.run_tournament()

test0 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/0.pt")
                  ,name="model_0")

test1 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/500.pt")
                  ,name="model_1")

# TODO: Delete, just testing
def agent_vs_mcts():
    for i in range(1000):
        player_to_move = 1
        game = HexWorld(size=configs.size)
        while True:
            if player_to_move == 2:
                # game = MonteCarlo(game, model=None).run()
                if type(game) is MonteCarloNode:
                    game = test0.perform_move_random(game.state)
                elif type(game) is HexWorld:
                    game = test0.perform_move_random(game)
                if game.is_final_state():
                    test1.losses += 1
                    print("RANDOM WON")
                    break
                player_to_move = 1
            elif player_to_move == 1:
                if type(game) is MonteCarloNode:
                    game = test1.perform_move_greedy(game.state)
                    # game = test0.perform_move_probabilistic(game.state)
                elif type(game) is HexWorld:
                    game = test1.perform_move_greedy(game)
                    # game = test0.perform_move_probabilistic(game)
                if game.is_final_state():
                    test1.wins += 1
                    print("AGENT WON")
                    break
                player_to_move = 2
    print(f'agent wins: {test1.wins}, agent losses: {test1.losses}')


# agent_vs_mcts()

