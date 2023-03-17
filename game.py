from abc import abstractmethod

import numpy as np

import configs

from typing import TypeVar

Game = TypeVar("Game", bound="Game")


class Game:
    def __init__(self, player: int = 1, size: int = configs.size):
        self.size = size
        self.player = player

    @abstractmethod
    def produce_initial_state(self) -> None:
        pass

    @abstractmethod
    def apply(self, action: str) -> Game:
        pass

    @abstractmethod
    def get_children_states(self) -> list[(str, Game)]:
        pass

    @abstractmethod
    def get_possible_actions(self) -> list[str]:
        pass

    @abstractmethod
    def is_final_state(self):
        pass

    @abstractmethod
    def enumerate_state(self) -> str:
        pass

    @abstractmethod
    def enumerate_state2(self) -> str:
        pass

    @abstractmethod
    def state_to_array(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_child_states_enumerated(self) -> list[str]:
        pass

    @abstractmethod
    def get_state_utility(self) -> int:
        pass
