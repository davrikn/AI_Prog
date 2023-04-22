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
from torch.nn import ModuleList

logger = getLogger()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class HexModel(Model):
    name = 'hex_v2'

    final_conv = 20

    def __init__(self, boardsize: int, snapshotdir: os.PathLike):
        super().__init__(boardsize, boardsize * boardsize, snapshotdir)
        self.conv1 = nn.Conv2d(2, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
        self.lin1 = nn.Linear(boardsize ** 2 * 64, 256)
        self.lin2 = nn.Linear(256, 128)
        self.lin3 = nn.Linear(128, boardsize ** 2)
        # final_out = self.init_model()
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

    def init_model(self):
        def resolve_activation_function(f: str):
            if f == 'linear':
                return F.linear
            elif f == 'sigmoid':
                return F.sigmoid
            elif f == 'tanh':
                return F.tanh
            elif f == 'relu':
                return F.relu
            else:
                raise Exception(f"Unknown activation function {f}")

        if len(configs.structure) == 0:
            configs.structure = [[128, 'relu']]

        last_outputs = 0
        modules = ModuleList()
        activation_funcs = []
        for i, conf in enumerate(configs.structure):
            inputs = self.size**2*self.final_conv if i == 0 else configs.structure[i-1][0]
            act = resolve_activation_function(conf[1])
            lin = nn.Linear(inputs, conf[0])
            modules.append(lin)
            activation_funcs.append(act)
            last_outputs = conf[0]
        self.activation_functions = activation_funcs
        self.linears = modules
        return last_outputs


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1)
        # for i in range(len(self.linears)):
        #     x = self.linears[i](x)
        #     x = self.activation_functions[i](x)
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
            pred = self(x).numpy()

            if p == -1:
                pred = self.transform_target(pred, k=1)

            actions = [(self.index_to_action[idx], probability) for idx, probability in enumerate(pred)]
            return sorted(actions, key=lambda tup: tup[1], reverse=True)

    def transform(self, x: np.ndarray) -> np.ndarray:
        _x1 = x[1].copy()
        x[1] = x[0]
        x[0] = _x1

        x[0] = np.rot90(x[0], -1)
        x[1] = np.rot90(x[1], -1)
        return x

    def transform_target(self, x: np.ndarray, k=1) -> np.ndarray:
        x = np.reshape(x, (self.size, self.size))
        x = np.rot90(x, k)
        return x.flatten()

    def train_batch(self, X: list[tuple[tuple[np.ndarray, int], list[tuple[str, float]]]]):
        self.optimizer.zero_grad()

        for x, _y in X:
            p = x[1]
            x = x[0]
            y = np.zeros(self.classes)

            for i, (k, v) in enumerate(_y):
                y[self.action_to_index[k]] = v

            if p == -1:
                x = self.transform(x)
                y = self.transform_target(y, k=-1)
            y = tensor(y, dtype=torch.float, requires_grad=False)
            x = tensor(x, dtype=torch.float, requires_grad=True)

            out = self(x)
            loss = self.LOSS_FUNCTION(out, y)
            loss.backward()
        self.optimizer.step()
        # print(f"\n\nY: {y.detach()}\nX: {x.detach()}\nOut: {out.detach()}\nLoss: {loss}")
