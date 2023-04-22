import random

import configs
from hex.hexAgent import HexAgent
from hex.hexModel import HexModel
from hex.hexWorld import HexWorld
from monteCarlo import MonteCarlo, MonteCarloNode
from tournament import Tournament

agent0 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + f"/hex_size_{configs.size}_checkpoint_0.pt")
                  , name="0")
agent25 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + f"/hex_size_{configs.size}_checkpoint_25.pt")
                   , name="25")
agent50 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + f"/hex_size_{configs.size}_checkpoint_50.pt")
                   , name="50")
# agent60 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + f"/hex_size_{configs.size}_checkpoint_60.pt")
#                    , name="Checkpoint60Agent")
agent100 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + f"/hex_size_{configs.size}_checkpoint_100.pt")
                    , name="100")
agent150 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + f"/hex_size_{configs.size}_checkpoint_150.pt")
                    , name="150")
agent200 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + f"/hex_size_{configs.size}_checkpoint_200.pt")
                    , name="200")
#
# agent250 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + f"/hex_size_{configs.size}_checkpoint_250.pt")
#                     , name="250")
# agent300 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + f"/hex_size_{configs.size}_checkpoint_300.pt")
#                     , name="300")
# agent350 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + f"/hex_size_{configs.size}_checkpoint_350.pt")
#                     , name="350")
# agent400 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + f"/hex_size_{configs.size}_checkpoint_400.pt")
#                     , name="400")
tournament = Tournament([agent0, agent25, agent50, agent100, agent150, agent200], G=configs.G, UI=False)
# tournament = Tournament([agent0, agent50, agent100, agent150, agent200], G=configs.G, UI=False)


tournament.run_tournament()

test100 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_4_checkpoint_150.pt")
                  ,name="model_0")
test0 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_4_checkpoint_0.pt")
                  ,name="model_0")

# TODO: Delete, just testing
def agent_vs_mcts():
    for i in range(500):
        player_to_move = 2
        game = HexWorld(size=configs.size)
        while True:
            if player_to_move == 1:
                # game = MonteCarlo(game, model=None).run()
                if type(game) is MonteCarloNode:
                    game = test100.perform_move_greedy(game.state)
                elif type(game) is HexWorld:
                    game = test100.perform_move_greedy(game)
                print(game, "\n")
                if game.is_final_state():
                    test100.wins += 1
                    print("TRAINED MODEL WON")
                    break
                player_to_move = 2
            elif player_to_move == 2:
                if type(game) is MonteCarloNode:
                    game = test0.perform_move_random(game.state)
                    # game = test100.perform_move_greedy(game.state)
                    # game = test0.perform_move_probabilistic(game.state)
                elif type(game) is HexWorld:
                    game = test0.perform_move_random(game)
                    # game = test100.perform_move_greedy(game)
                    # game = test0.perform_move_probabilistic(game)
                print(game, "\n")
                if game.is_final_state():
                    test100.losses += 1
                    print("RANDOM WON")
                    break
                player_to_move = 1
    print(f'TRAINED MODEL wins: {test100.wins}, TRAINED MODEL losses: {test100.losses}')


# agent_vs_mcts()

