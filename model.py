from abc import abstractmethod
import numpy as np
import torch.nn as nn

class Model(nn.Module):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def classify(self, x: np.ndarray) -> list[str]:
        pass