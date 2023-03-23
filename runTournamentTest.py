import configs
from hex.hexAgent import HexAgent
from hex.hexModel import HexModel
from tournament import Tournament

agent0 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_4_checkpoint_0.pt")
                  , name="Checkpoint 0 agent")
agent50 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_4_checkpoint_50.pt")
                   , name="Checkpoint 50 agent")
agent100 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_4_checkpoint_100.pt")
                    , name="Checkpoint 100 agent")
agent150 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_4_checkpoint_150.pt")
                    , name="Checkpoint 150 agent")
agent200 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_4_checkpoint_200.pt")
                    , name="Checkpoint 200 agent")

tournament = Tournament([agent0, agent50, agent100, agent150, agent200])
tournament.run_tournament()
