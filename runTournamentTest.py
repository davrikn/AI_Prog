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
agent75 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + f"/hex_size_{configs.size}_checkpoint_75.pt")
                   , name="75")
# agent60 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + f"/hex_size_{configs.size}_checkpoint_60.pt")
#                    , name="Checkpoint60Agent")
agent100 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + f"/hex_size_{configs.size}_checkpoint_100.pt")
                    , name="100")
# agent125 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + f"/hex_size_{configs.size}_checkpoint_125.pt")
#                     , name="125")
agent150 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + f"/hex_size_{configs.size}_checkpoint_150.pt")
                    , name="150")
agent200 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + f"/hex_size_{configs.size}_checkpoint_200.pt")
                    , name="200")
agent225 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + f"/hex_size_{configs.size}_checkpoint_225.pt")
                    , name="225")
agent250 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + f"/hex_size_{configs.size}_checkpoint_250.pt")
                    , name="250")
agent300 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + f"/hex_size_{configs.size}_checkpoint_300.pt")
                    , name="300")
# agent350 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + f"/hex_size_{configs.size}_checkpoint_350.pt")
#                     , name="350")
# agent400 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + f"/hex_size_{configs.size}_checkpoint_400.pt")
#                     , name="400")
tournament = Tournament([agent0, agent50, agent100, agent150, agent200, agent250, agent300], G=configs.G, UI=False)
# tournament = Tournament([agent0, agent50, agent100, agent150, agent200], G=configs.G, UI=False)


# tournament.run_tournament()

trained_model = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_7_checkpoint_150.pt")
                         , name="trained")
random_model = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_7_checkpoint_0.pt")
                        , name="random")

# TODO: Delete, just testing
def agent_vs_mcts():
    for i in range(10):
        player_to_move = 1
        game = HexWorld(size=configs.size)
        while True:
            if player_to_move == 1:
                # game = MonteCarlo(game, model=None).run()
                if type(game) is MonteCarloNode:
                    game = trained_model.perform_move_greedy(game.state)
                elif type(game) is HexWorld:
                    game = trained_model.perform_move_greedy(game)
                print(game.state(deNested=True), "\n")
                if game.is_final_state():
                    # print(game.state(deNested=True), "\n")
                    trained_model.wins += 1
                    print("TRAINED MODEL WON")
                    break
                player_to_move = 2
            elif player_to_move == 2:
                if type(game) is MonteCarloNode:
                    game = random_model.perform_move_random(game.state)
                    # game = test100.perform_move_greedy(game.state)
                    # game = test0.perform_move_probabilistic(game.state)
                elif type(game) is HexWorld:
                    game = random_model.perform_move_random(game)
                    # game = test100.perform_move_greedy(game)
                    # game = test0.perform_move_probabilistic(game)
                # print(game.state(deNested=True), "\n")
                if game.is_final_state():
                    trained_model.losses += 1
                    print("RANDOM WON")
                    break
                player_to_move = 1
    print(f'TRAINED MODEL wins: {trained_model.wins}, TRAINED MODEL losses: {trained_model.losses}')


agent_vs_mcts()

