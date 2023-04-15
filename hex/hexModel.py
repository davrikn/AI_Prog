import copy

import configs
from model import Model
import numpy as np
from torch import nn, load, tensor
from os.path import isfile
import os
import torch
from logging import getLogger
import torch.nn.functional as F


logger = getLogger()


class HexModel(Model):
    name = 'hex_v2'

    def __init__(self, boardsize: int, snapshotdir: os.PathLike = '/'):
        super().__init__(boardsize, boardsize * boardsize, snapshotdir)
        self.lin1 = nn.Linear(boardsize**2*2, 64)
        self.lin2 = nn.Linear(64, 128)
        self.lin3 = nn.Linear(128, boardsize * boardsize)
        self.sm = nn.Softmax(dim=0)
        self.action_to_index = self.gen_action_index_dict()
        self.index_to_action = {v: k for k, v in self.action_to_index.items()}
        self.action_to_index_transpose = self.gen_action_index_dict_transpose()
        self.index_to_action_transpose = {v: k for k, v in self.action_to_index_transpose.items()}

        self.crit = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=configs.learning_rate)

        if isfile(f"{snapshotdir}"):
            logger.info("Loading statedict")
            self.load_state_dict(load(f"{snapshotdir}"))
            logger.info("Finished loading statedict")

        if isfile(f"{snapshotdir}/{self.name}_size_{boardsize}.pth"):
            logger.info("Loading statedict")
            self.load_state_dict(load(f"{snapshotdir}/{self.name}_size_{boardsize}.pth"))
            logger.info("Finished loading statedict")

    def pad(self, input: str or int, length: int = 2, start=True):
        padding = "0" * length
        out = padding + str(input) if start else str(input) + padding
        return out[-length:]

    def gen_action_index_dict(self):
        action_to_index = dict()
        k = 0
        for i in range(self.size):
            for j in range(self.size):
                action = self.pad(i) + self.pad(j)
                action_to_index[action] = k
                k += 1
        return action_to_index

    def gen_action_index_dict_transpose(self):
        action_to_index_transpose = dict()
        k = 0
        for i in range(self.size):
            for j in range(self.size):
                action = self.pad(j) + self.pad(i)
                action_to_index_transpose[action] = k
                k += 1
        return action_to_index_transpose

    def transform(self, x: np.ndarray) -> np.ndarray:
        _x1 = x[1].copy()
        x[1] = x[0]
        x[0] = _x1

        x[0] = np.rot90(x[0], -1)
        x[1] = np.rot90(x[1], -1)
        return x

    def transform_target(self, x: np.ndarray) -> np.ndarray:
        x = np.reshape(x, (self.size, self.size))
        x = np.rot90(x, -1)
        return x.flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = self.sm(x)
        return x

    def classify(self, x: tuple[np.ndarray, int]) -> list[tuple[str, float]]:
        with torch.no_grad():
            p = x[1]
            x = x[0]
            if p == -1:
                x = self.transform(x)
            x = tensor(x, dtype=torch.float)
            x = self(x).numpy()
            if p == -1:
                x = self.transform_target(x)
            actions = [(self.index_to_action[idx], probability) for idx, probability in enumerate(x)]
            return sorted(actions, key=lambda tup: tup[1])

    def train_batch(self, X: list[tuple[tuple[np.ndarray, int], list[tuple[str, float]]]]):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

        print(f"Batch len: {len(X)}")
        for x, _y in X:

            p = x[1]

            if p == -1:
                continue

            x = x[0]

            y = np.zeros(self.classes)
            for k, v in _y:
                y[self.action_to_index[k]] = v

            if p == -1:
                x = self.transform(x)
                y = self.transform_target(y)

            y = tensor(y, dtype=torch.float, requires_grad=False)
            x = tensor(x, dtype=torch.float, requires_grad=True)


            optimizer.zero_grad()
            out = self(x)
            print(f"\n\nY:{y}\nX:{x}\nOut:{out}")

            loss = self.crit(out, y)
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    model = HexModel(2)
    c = model.classify((np.array([[[1,0],[0,0]],[[0,1],[0,0]]]), 1))
    print(c)