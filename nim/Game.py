from abc import abstractmethod


class Game:

    @abstractmethod
    def produce_initial_state(self):
        pass

    @abstractmethod
    def get_children_states(self):
        pass

    @abstractmethod
    def is_final_state(self):
        pass

