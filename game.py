from abc import abstractmethod


class Game:
    def __init__(self, size: int):
        self.size = size

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