import random

import configs
from hex.hexAgent import HexAgent
from hex.hexModel import HexModel
from hex.hexWorld import HexWorld
from monteCarlo import MonteCarlo, MonteCarloNode
from tournament import Tournament

test0 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/0.pt"), name="__________0")
test50 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/250.pt"), name="__________250")
test100 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/500.pt"), name="__________500")
test150 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/750.pt"), name="__________750")
test200 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/1000.pt"), name="__________1000")

print(list(test0.model.parameters()))
print(list(test50.model.parameters()))
print(list(test100.model.parameters()))
print(list(test150.model.parameters()))
print(list(test200.model.parameters()))

# tournament = Tournament([agent0, agent50, agent100, agent150, agent200], UI=False)
tournament = Tournament([test0, test50, test100, test150, test200], G=99, UI=True)


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

