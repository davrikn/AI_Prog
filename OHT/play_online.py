# Import and initialize your own actor
import configs
from hex.hexAgent import HexAgent
from ActorClient import ActorClient
from hex.hexModel import HexModel

actor = HexAgent(model=HexModel(boardsize=configs.size,
                                snapshotdir=configs.model_dir + 'hex_size_7_checkpoint_0.pt'), name="OHT")


# Import and override the `handle_get_action` hook in ActorClient


class MyClient(ActorClient):
    player_to_move = 0

    def handle_game_start(self, start_player):
        MyClient.player_to_move = start_player

    def handle_get_action(self, state):
        row, col = actor.get_action(state, MyClient.player_to_move)  # Your logic
        return row, col


# Initialize and run your overridden client when the script is executed
if __name__ == '__main__':
    client = MyClient(auth='<your token>', qualify=False)
    client.run()
