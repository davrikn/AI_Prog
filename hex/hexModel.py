import os.path

import numpy as np
from torch import nn, load
from os.path import isfile

class HexModel(nn.Module):
    def __init__(self, boardsize: int):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, 1, 1)
        self.mp1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 96, 3, 1, 1)
        self.mp2 = nn.MaxPool2d(2)
        self.lin1 = nn.Linear(96*boardsize*boardsize, 256)
        self.lin2 = nn.Linear(256, 128)
        self.lin3 = nn.Linear(1, boardsize*boardsize + 1)
        self.sm = nn.Softmax(boardsize * boardsize + 1)

        if isfile('../model_dicts/hex.pth'):
            self.load_state_dict(load('../model_dicts/hex.pth'))


    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.conv1(x)
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.mp1(x)
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
        x = self.sm(x)
        return x

    def classify(self, x) -> int:
        return np.argmax(self.forward(x), 0)