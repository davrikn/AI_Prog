import copy

import configs
from model import Model
import numpy as np
from torch import nn, load, tensor
from os.path import isfile
import os
import torch
from logging import getLogger
import random
from torchmetrics import Accuracy

logger = getLogger()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class HexModel(Model):
    name = 'hex_v2'

    def __init__(self, boardsize: int, snapshotdir: os.PathLike):
        super().__init__(boardsize, boardsize * boardsize, snapshotdir)
        self.conv1 = nn.Conv2d(2, 30, 3, 1, 1)
        self.conv2 = nn.Conv2d(30, 20, 3, 1, 1)
        self.seq = self.init_model()
        self.sm = nn.Softmax(dim=0)
        self.action_to_index = self.gen_action_index_dict()
        self.index_to_action = {v: k for k, v in self.action_to_index.items()}
        self.action_to_index_transpose = self.gen_action_index_dict_transpose()
        self.index_to_action_transpose = {v: k for k, v in self.action_to_index_transpose.items()}

        self.optimizer = self.resolve_optimizer()

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
        if x[1] == -1:
            temp = copy.deepcopy(x[0][0])
            x[0][0] = x[0][1]
            x[0][1] = temp
            x[0][0] = np.rot90(x[0][0], k=-1)
            x[0][1] = np.rot90(x[0][1], k=-1)
            # x[0][0] = np.transpose(x[0][0])
            # x[0][1] = np.transpose(x[0][1])


    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x = self.conv1(x[0])
        x = self.conv2(x)
        x = x.view(-1)
        for i in range(len(configs.structure)*2):
            x = self.linears[i](x)
        x = self.sm(x)
        return x

    def classify(self, x: tuple[np.ndarray, int]) -> list[tuple[str, float]]:
        _player = x[1]
        self.preprocess(x)
        x = tensor(x[0], dtype=torch.float, device=device), tensor([x[1]], dtype=torch.float)
        x = self(x)
        if _player == -1:
            x = self.transpose_actions(x)
            # x = x.detach().numpy()
            # actions = [(self.index_to_action_transpose[idx], probability) for idx, probability in enumerate(x)]

        x = x.detach().numpy()
        actions = [(self.index_to_action[idx], probability) for idx, probability in enumerate(x)]
        # sorted_actions = [x for _, x in sorted(zip(x, actions), key=lambda pair: pair[0], reverse=True)]
        # return list(map(lambda x: self.index_to_action[x], np.argsort(x)))
        # return sorted_actions
        return sorted(actions, key=lambda tup: tup[1])

    def transpose_actions(self, x, k=1):
        x = x.view(self.size, self.size)
        x = x.detach().numpy()
        x = np.rot90(x, k)
        x = x.flatten()

        return torch.tensor(x, dtype=torch.float, requires_grad=True)


    def train_batch(self, X: list[tuple[tuple[np.ndarray, int], list[tuple[str, float]]]]):
        random.shuffle(X)
        for x in X:
            self.preprocess(x[0])
        accuracies = []
        acc = Accuracy(task="multiclass", num_classes=self.classes)
        for i, (_x, _y) in enumerate(X, 1):
            if i % 100 == 0:
                logger.debug(f"Trained on {i} samples")
            self.optimizer.zero_grad()
            y = np.zeros(self.classes)
            for k, v in _y:
                y[self.action_to_index[k]] = v
            y = torch.tensor(y, dtype=torch.float, requires_grad=True, device=device)
            x = torch.tensor(_x[0], dtype=torch.float, requires_grad=True, device=device), torch.tensor([_x[1]], dtype=torch.float)
            out = self(x)
            if _x[1] == -1:
                y = self.transpose_actions(y, k=-1)

            loss = self.LOSS_FUNCTION(out, y)
            loss.backward()
            self.optimizer.step()
            print(f"\n\nY: {y.detach()}\nX: {x}\nOut: {out.detach()}\nLoss: {loss}")
            accuracies.append(acc(out, y).item())

        tot_acc = 0
        for accuracy in accuracies:
            tot_acc += accuracy
        return tot_acc/len(accuracies)
