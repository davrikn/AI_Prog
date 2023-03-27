import logging
import random
from abc import abstractmethod
from os.path import exists

import numpy as np
import torch.nn as nn
import torch
import os

import configs
import utils

logger = logging.getLogger()

class Model(nn.Module):
    rbuf: list[tuple[tuple[np.ndarray, int], list[tuple[str, float]]]] = []
    name = 'model'
    crit = nn.CrossEntropyLoss()
    action_to_index: dict[str, int]
    index_to_action: dict[int, str]
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
    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        pass

    @abstractmethod
    def classify(self, x: tuple[np.ndarray, int]) -> list[str]:
        pass

    @abstractmethod
    def gen_action_to_index(self) -> dict[str, int]:
        pass

    def train_batch(self, X: list[tuple[tuple[np.ndarray, int], list[tuple[str, float]]]]):
        for x in X:
            self.preprocess(x[0])
        for i, (_x, _y) in enumerate(X, 1):
            if i % 100 == 0:
                logger.debug(f"Trained on {i} samples")
            self.optimizer.zero_grad()
            y = np.zeros(self.classes)
            for k, v in _y:
                y[self.action_to_index[k]] = v
            y = torch.tensor(y, dtype=torch.float)
            x = torch.tensor(_x[0], dtype=torch.float, requires_grad=True), torch.tensor([_x[1]], dtype=torch.float)
            x = self(x)

            x2 = x.detach()

            for idx, prob in enumerate(y):
                if prob == 0:
                    x2[idx] = 0
            scaled_x_only_valid = x2.numpy() / np.sum(x2.numpy())
            for idx, scaled_x in enumerate(scaled_x_only_valid):
                x2[idx] = float(scaled_x)

            x2 = torch.tensor(x2.numpy(), dtype=torch.float, requires_grad=True)

            loss = self.crit(x2, y)
            loss.backward()
            self.optimizer.step()

    def append_rbuf(self, data: list[tuple[tuple[np.ndarray, int], list[tuple[str, float]]]]):
        self.rbuf.extend(data)

    def append_rbuf_single(self, data: tuple[tuple[np.ndarray, int], list[tuple[str, float]]]):
        self.rbuf.append(data)

    def flush_rbuf(self):
        random.shuffle(self.rbuf)
        utils.save_train_data(self.rbuf)

        self.train_batch(self.rbuf)
        logging.info("Training batch")
        self.rbuf = []
        logging.debug("Saving statedict")
        # torch.save(self.state_dict(), f"{self.snapshotdir}/{self.name}_size_{self.size}.pth")
        logging.debug("Finished saving statedict")

    def save_model(self, file_name: str):
        torch.save(self.state_dict(), f"{configs.model_dir}/{file_name}.pt")

