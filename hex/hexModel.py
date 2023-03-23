import configs
from model import Model
import numpy as np
from torch import nn, load, tensor
from os.path import isfile
import os
import torch
from logging import getLogger

logger = getLogger()

class HexModel(Model):
    name = 'hex_v2'

    def __init__(self, boardsize: int, snapshotdir: os.PathLike):
        super().__init__(boardsize, boardsize*boardsize, snapshotdir)
        self.conv1 = nn.Conv2d(2, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.lin1 = nn.Linear(64 * boardsize * boardsize, 128)
        self.lin2 = nn.Linear(128, boardsize * boardsize)
        # self.conv1 = nn.Conv2d(2, 32, 3, 1, 1)
        # self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        # self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        # self.lin1 = nn.Linear(128*boardsize*boardsize, 512)
        # self.lin2 = nn.Linear(512, 256)
        # self.lin3 = nn.Linear(256, boardsize*boardsize)
        self.sm = nn.Softmax(dim=0)
        self.action_to_index = self.gen_action_index_dict()
        self.index_to_action = {v: k for k, v in self.action_to_index.items()}

        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

        if isfile(f"{snapshotdir}"):
            logger.info("Loading statedict")
            self.load_state_dict(load(f"{snapshotdir}"))
            logger.info("Finished loading statedict")

        if isfile(f"{snapshotdir}/{self.name}_size_{boardsize}.pth"):
            logger.info("Loading statedict")
            self.load_state_dict(load(f"{snapshotdir}/{self.name}_size_{boardsize}.pth"))
            logger.info("Finished loading statedict")

    def pad(self, input: str or int, length: int = 2, start=True):
        padding = "0"*length
        out = padding+str(input) if start else str(input)+padding
        return out[-length:]

    def gen_action_index_dict(self):
        action_to_index = dict()
        k = 0
        for i in range(self.size):
            for j in range(self.size):
                action = self.pad(i)+self.pad(j)
                action_to_index[action] = k
                k += 1
        return action_to_index

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # TODO: experiment with flipping channels on antagonous player
        if x[1] == -1:
            temp = x[0][0]
            x[0][0] = x[0][1]
            x[0][1] = temp

        x = self.conv1(x[0])
        #x = self.mp1(x)
        x = self.conv2(x)
        #x = self.mp2(x)
        # x = self.conv3(x)
        x = x.view(-1)
        x = self.lin1(x)
        x = self.lin2(x)
        # x = self.lin3(x)
        return x

    def classify(self, x: tuple[np.ndarray, int]) -> list[str]:
        x = tensor(x[0], dtype=torch.float), tensor([x[1]], dtype=torch.float)
        x = self(x)
        x = self.sm(x)
        x = x.detach().numpy()
        return list(map(lambda x: self.index_to_action[x], np.argsort(x)))
