import copy

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
        super().__init__(boardsize, boardsize * boardsize, snapshotdir)
        self.n1_conv1 = nn.Conv2d(2, 32, 3, 1, 1)
        self.n1_lin1 = nn.Linear(boardsize**2*32, 2048)
        self.n1_lin2 = nn.Linear(2048, boardsize * boardsize)
        self.n2_conv1 = nn.Conv2d(2, 32, 3, 1, 1)
        self.n2_lin1 = nn.Linear(boardsize**2*32, 2048)
        self.n2_lin2 = nn.Linear(2048, boardsize * boardsize)
        self.sm = nn.Softmax(dim=0)
        self.action_to_index = self.gen_action_index_dict()
        self.index_to_action = {v: k for k, v in self.action_to_index.items()}
        self.action_to_index_transpose = self.gen_action_index_dict_transpose()
        self.index_to_action_transpose = {v: k for k, v in self.action_to_index_transpose.items()}

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

    def preprocess(self, x: tuple[np.ndarray, int]) -> None:
        pass


    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if x[1] == 1:
            x = self.n1_conv1(x[0])
            x = x.view(-1)
            x = self.n1_lin1(x)
            x = self.n1_lin2(x)
        else:
            x = self.n2_conv1(x[0])
            x = x.view(-1)
            x = self.n2_lin1(x)
            x = self.n2_lin2(x)
        return x

    def classify(self, x: tuple[np.ndarray, int]) -> list[tuple[str, float]]:
        _player = x[1]
        self.preprocess(x)
        x = tensor(x[0], dtype=torch.float), tensor([x[1]], dtype=torch.float)
        x = self(x)

        x = x.detach().numpy()
        actions = [(self.index_to_action[idx], probability) for idx, probability in enumerate(x)]
        return sorted(actions, key=lambda tup: tup[1])

    def train_batch(self, X: list[tuple[tuple[np.ndarray, int], list[tuple[str, float]]]]):
        for x in X:
            self.preprocess(x[0])
        epochs = 3
        weights_pre = [x.clone() for x in self.parameters()]
        for epoch in range(epochs):
            for i, (_x, _y) in enumerate(X, 1):
                if i % 100 == 0:
                    logger.debug(f"Trained on {i} samples")
                self.optimizer.zero_grad()
                y = np.zeros(self.classes)
                for k, v in _y:
                    y[self.action_to_index[k]] = v
                y = torch.tensor(y, dtype=torch.float, requires_grad=True)
                x = torch.tensor(_x[0], dtype=torch.float, requires_grad=True), torch.tensor([_x[1]], dtype=torch.float)
                x = self(x)

                loss = self.LOSS_FUNCTION(x, y)
                loss.backward()
                self.optimizer.step()
        if False not in [torch.equal(weights_pre[i], list(self.parameters())[i]) for i in range(len(weights_pre))]:
            print("Weights unchanged")
        else:
            pass
            #print(weights_pre[0] - list(self.parameters())[0])
