from abc import abstractmethod

import numpy as np

import configs

from typing import TypeVar, Union, overload

Game = TypeVar("Game", bound="Game")


class Game:
    def __init__(self, player: int = 1, size: int = configs.size):
        self.size = size
        self.player = player

    @abstractmethod
    def get_children_states(self) -> list[(str, Game)]:
        pass

    @abstractmethod
    def get_possible_actions(self) -> list[str]:
        pass

    @abstractmethod
    def apply(self, action: str, deepcopy: bool) -> Game:
        pass

    @abstractmethod
    def state_stringified(self) -> str:
        pass

    @abstractmethod
    def state(self, deNested: bool = False) -> tuple[np.ndarray, int]:
        pass

    @abstractmethod
    def is_final_state(self) -> bool:
        pass

    @abstractmethod
    def get_utility(self) -> int:
        pass

    @abstractmethod
    def reset(self) -> Game:
        pass