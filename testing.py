import configs
from hex.hexModel import HexModel
from hex.hexWorld import HexWorld

model = HexModel(configs.size, snapshotdir=configs.model_dir + "/1000.pt")

game = HexWorld(configs.size)

s = game.state(True)

out = model.classify(s)

print(out)