from abc import abstractmethod

import configs


class Game:
    def __init__(self, state=None, size: int = configs.size):
        self.size = size
        self.state = state

    @abstractmethod
    def produce_initial_state(self):
        pass

    @abstractmethod
    def get_children_states(self):
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
    def get_child_states_enumerated(self) -> list[str]:
        pass
