# Import and initialize your own actor
import numpy as np

import configs
from hex.hexAgent import HexAgent
from ActorClient import ActorClient
from hex.hexModel import HexModel

actor = HexAgent(model=HexModel(boardsize=7,
                                snapshotdir=configs.model_dir + 'hex_size_7_checkpoint_125.pt'), name="OHT")


# Import and override the `handle_get_action` hook in ActorClient


class MyClient(ActorClient):
    starting_player = 0

    def handle_game_start(self, start_player):
        if start_player == 1:
            MyClient.starting_player = 1
        else:
            MyClient.starting_player = -1

    def handle_get_action(self, state):
        row, col = actor.get_action(state, MyClient.starting_player)  # Your logic
        return row, col

    def handle_game_over(self, winner, end_state):
        player_id = end_state.pop(0)
        end_state = np.reshape(end_state, (configs.OHT_size, configs.OHT_size))
        if winner == 1:
            print("We won (1)")
        else:
            print("OHT won (2)")
        if MyClient.starting_player == 1:
            print("We started")
        else:
            print("OHT started")
        print("END STATE:")
        print(end_state, "\n\n")


# Initialize and run your overridden client when the script is executed
if __name__ == '__main__':
    client = MyClient(auth='', qualify=False)
    client.run()
