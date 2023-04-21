import random

import configs
from hex.hexAgent import HexAgent
from hex.hexModel import HexModel
from hex.hexWorld import HexWorld
from monteCarlo import MonteCarlo, MonteCarloNode
from tournament import Tournament

agent0 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + f"/hex_size_{configs.size}_checkpoint_0.pt")
                  , name="Checkpoint0Agent")
agent10 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + f"/hex_size_{configs.size}_checkpoint_10.pt")
                   , name="Checkpoint10Agent")
agent20 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + f"/hex_size_{configs.size}_checkpoint_20.pt")
                   , name="Checkpoint20Agent")
agent40 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + f"/hex_size_{configs.size}_checkpoint_40.pt")
                   , name="Checkpoint40Agent")
agent50 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + f"/hex_size_{configs.size}_checkpoint_50.pt")
                   , name="Checkpoint50Agent")
# agent60 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + f"/hex_size_{configs.size}_checkpoint_60.pt")
#                    , name="Checkpoint60Agent")
agent100 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + f"/hex_size_{configs.size}_checkpoint_100.pt")
                    , name="Checkpoint100Agent")
agent150 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + f"/hex_size_{configs.size}_checkpoint_150.pt")
                    , name="Checkpoint150Agent")
# agent200 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + f"/hex_size_{configs.size}_checkpoint_200.pt")
#                     , name="Checkpoint200Agent")
tournament = Tournament([agent0, agent50, agent100], G=configs.G, UI=False)
# tournament = Tournament([agent0, agent50, agent100, agent150, agent200], G=configs.G, UI=False)


tournament.run_tournament()

test100 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_4_checkpoint_100.pt")
                  ,name="model_0")
test0 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_4_checkpoint_0.pt")
                  ,name="model_0")

# TODO: Delete, just testing
def agent_vs_mcts():
    for i in range(25):
        player_to_move = 1
        game = HexWorld(size=configs.size)
        while True:
            if player_to_move == 2:
                # game = MonteCarlo(game, model=None).run()
                if type(game) is MonteCarloNode:
                    game = test100.perform_move_greedy(game.state)
                elif type(game) is HexWorld:
                    game = test100.perform_move_greedy(game)
                print(game, "\n")
                if game.is_final_state():
                    test0.losses += 1
                    print("MCTS WON")
                    break
                player_to_move = 1
            elif player_to_move == 1:
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
                    test0.wins += 1
                    print("RANDOM WON")
                    break
                player_to_move = 2
    print(f'Random wins: {test0.wins}, Random losses: {test0.losses}')


# agent_vs_mcts()

