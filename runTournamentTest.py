import random

import configs
from hex.hexAgent import HexAgent
from hex.hexModel import HexModel
from hex.hexWorld import HexWorld
from monteCarlo import MonteCarlo, MonteCarloNode
from tournament import Tournament

agent0 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_4_checkpoint_0.pt")
                  , name="Checkpoint0Agent")
agent50 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_4_checkpoint_50.pt")
                   , name="Checkpoint50Agent")
agent100 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_4_checkpoint_100.pt")
                    , name="Checkpoint100Agent")
agent150 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_4_checkpoint_150.pt")
                    , name="Checkpoint150Agent")
agent200 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_4_checkpoint_200.pt")
                    , name="Checkpoint200Agent")
agent250 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_4_checkpoint_250.pt")
                    , name="Checkpoint250Agent")
agent300 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_4_checkpoint_300.pt")
                    , name="Checkpoint300Agent")
agent400 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_4_checkpoint_400.pt")
                    , name="Checkpoint400Agent")
agent500 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_4_checkpoint_500.pt")
                    , name="Checkpoint500Agent")
# agent550 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_4_checkpoint_550.pt")
#                     , name="Checkpoint550Agent")
agent600 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_4_checkpoint_600.pt")
                    , name="Checkpoint600Agent")
agent700 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_4_checkpoint_700.pt")
                    , name="Checkpoint700Agent")
agent800 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_4_checkpoint_800.pt")
                    , name="Checkpoint800Agent")
agent900 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_4_checkpoint_900.pt")
                    , name="Checkpoint900Agent")
agent1000 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_4_checkpoint_1000.pt")
                    , name="Checkpoint1000Agent")
#

tournament = Tournament([agent0, agent50, agent100, agent150, agent200, agent300, agent400], G=configs.G, UI=False)
# tournament = Tournament([test200, test300, test400, test0, test1000, test100], G=150, UI=True)

tournament.run_tournament()

test0 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_4_checkpoint_0.pt")
                  ,name="model_0")

# TODO: Delete, just testing
def agent_vs_mcts():
    for i in range(25):
        player_to_move = 1
        game = HexWorld(size=configs.size)
        while True:
            if player_to_move == 2:
                game = MonteCarlo(game, model=None).run()
                print(game.state, "\n")
                # if type(game) is MonteCarloNode:
                #     game = test0.perform_move_random(game.state)
                # elif type(game) is HexWorld:
                #     game = test0.perform_move_random(game)
                if game.state.is_final_state():
                    test0.losses += 1
                    print("MCTS WON")
                    break
                player_to_move = 1
            elif player_to_move == 1:
                if type(game) is MonteCarloNode:
                    game = test0.perform_move_random(game.state)
                    # game = test0.perform_move_probabilistic(game.state)
                elif type(game) is HexWorld:
                    game = test0.perform_move_random(game)
                    # game = test0.perform_move_probabilistic(game)
                print(game, "\n")
                if game.is_final_state():
                    test0.wins += 1
                    print("RANDOM WON")
                    break
                player_to_move = 2
    print(f'Random wins: {test0.wins}, Random losses: {test0.losses}')


# agent_vs_mcts()

