
import configs
from hex.hexAgent import HexAgent
from hex.hexModel import HexModel
from tournament import Tournament


agent0 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_5_checkpoint_0.pt")
                  , name="Checkpoint0Agent")
agent50 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_5_checkpoint_50.pt")
                   , name="Checkpoint50Agent")
agent100 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_5_checkpoint_100.pt")
                    , name="Checkpoint100Agent")
agent150 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_5_checkpoint_150.pt")
                    , name="Checkpoint150Agent")
agent200 = HexAgent(HexModel(boardsize=configs.size, snapshotdir=configs.model_dir + "/hex_size_5_checkpoint_200.pt")
                    , name="Checkpoint200Agent")

tournament = Tournament([agent0, agent50, agent100, agent150, agent200], UI=False)
tournament.run_tournament()
