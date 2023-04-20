import logging
import random
from abc import abstractmethod
from os.path import exists

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import os

import configs
import utils

logger = logging.getLogger()

class Model(nn.Module):
    rbuf: list[tuple[tuple[np.ndarray, int], list[tuple[str, float]]]] = []
    name = 'model'
    # crit = nn.CrossEntropyLoss()
    LOSS_FUNCTION = configs.loss_function
    action_to_index: dict[str, int]
    index_to_action: dict[int, str]
    # optimizer: torch.optim.Optimizer
    optimizer: torch.optim.Optimizer

    def __init__(self, size: int, classes: int, snapshotdir: os.PathLike):
        super().__init__()
        self.size = size
        self.classes = classes
        self.snapshotdir = snapshotdir

    @abstractmethod
    def preprocess(self, x: tuple[np.ndarray, int]) -> torch.Tensor:
        pass

    @abstractmethod
    def transpose_actions(self, x: torch.Tensor, k) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        pass

    @abstractmethod
    def classify(self, x: tuple[np.ndarray, int]) -> list[tuple[str, float]]:
        pass

    @abstractmethod
    def gen_action_to_index(self) -> dict[str, int]:
        pass

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

        modules = dict()
        for i, conf in enumerate(configs.structure):
            modules[i*2] = nn.Linear(conf[0], conf[1])
            modules[i*2+1] = resolve_activation_function(conf[2])
        self.linears = modules


    def train_batch(self, X: list[tuple[tuple[np.ndarray, int], list[tuple[str, float]]]]):
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


    def resolve_optimizer(self) -> torch.optim.Optimizer:
        if configs.optimizer == 'adagrad':
            return torch.optim.Adagrad(self.parameters(), lr=configs.learning_rate)
        elif configs.optimizer == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=configs.learning_rate, momentum=0.9)
        elif configs.optimizer == 'rmsprop':
            return torch.optim.RMSprop(self.parameters(), lr=configs.learning_rate)
        elif configs.optimizer == 'adam':
            return torch.optim.Adam(self.parameters(), lr=configs.learning_rate)
        else:
            raise Exception("Unknown optimizer")

    def append_rbuf(self, data: list[tuple[tuple[np.ndarray, int], list[tuple[str, float]]]]):
        self.rbuf = data + self.rbuf
        if len(self.rbuf) > 250:
            self.rbuf = self.rbuf[:250]

    def append_rbuf_single(self, data: tuple[tuple[np.ndarray, int], list[tuple[str, float]]]):
        self.rbuf.insert(0, data)
        if len(self.rbuf) > 250:
            self.rbuf = self.rbuf[:250]

    def flush_rbuf(self):
        if configs.save_data:
            utils.save_train_data(self.rbuf)

        logging.info("Training batch")
        batchsize = 10 if len(self.rbuf) > 10 else len(self.rbuf)
        for i in range(configs.epochs):
            self.train_batch(random.sample(self.rbuf, batchsize))

    def save_model(self, file_name: str):
        torch.save(self.state_dict(), f"{configs.model_dir}/{file_name}.pt")

    def remove_invalid_moves(self, x, y):
        x2 = x.detach()

        for idx, prob in enumerate(y):
            if prob == 0:
                x2[idx] = 0
        scaled_x_only_valid = x2.numpy() / np.sum(x2.numpy())
        for idx, scaled_x in enumerate(scaled_x_only_valid):
            x[idx] = float(scaled_x)

        return torch.tensor(x2.numpy(), dtype=torch.float, requires_grad=True)

