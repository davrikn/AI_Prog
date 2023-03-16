import logging
from abc import abstractmethod
import numpy as np
import torch.nn as nn
import torch
import os

import configs

logger = logging.getLogger()
logger.setLevel(configs.log_level)

class Model(nn.Module):
    rbuf: list[tuple[np.ndarray, list[tuple[str, float]]]] = []
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
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def classify(self, x: np.ndarray) -> list[str]:
        pass

    @abstractmethod
    def gen_action_to_index(self) -> dict[str, int]:
        pass

    def train_batch(self, X: list[tuple[np.ndarray, list[tuple[str, float]]]]):
        for i, (_x, _y) in enumerate(X):
            if i % 100 == 0:
                logger.info(f"Trained on {i} samples")
            self.optimizer.zero_grad()
            y = np.zeros(self.classes)
            for k, v in _y:
                y[self.action_to_index[k]] = v
            y = torch.tensor(y, dtype=torch.float)

            x = torch.tensor(_x, dtype=torch.float)
            x = self(x)

            loss = self.crit(x, y)
            loss.backward()
            self.optimizer.step()

    def append_rbuf(self, data: list[tuple[np.ndarray, list[tuple[str, float]]]]):
        self.rbuf.extend(data)

    def append_rbuf_single(self, data: tuple[np.ndarray, list[tuple[str, float]]]):
        self.rbuf.append(data)

    def flush_rbuf(self):
        self.train_batch(self.rbuf)
        self.rbuf = []
        torch.save(self.state_dict(), f"{self.snapshotdir}/{self.name}_size_{self.size}.pth")