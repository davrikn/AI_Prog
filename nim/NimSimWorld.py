from nim.Game import Game
import copy


class NimSimWorld(Game):
    def __init__(self, size: int = 3):
        self.size = size
        self.board = self.__init_board()

    def produce_initial_state(self):
        return self.board

    def get_children_states(self):
        return self.__generate_all_child_states()

    def is_final_state(self):
        return self.__check_final_state()

    def __init_board(self):
        board = []
        for i in range(self.size):
            board.append(i + 1)
        return board

    def pick_object_from_pile(self, pile):
        self.board[pile] -= 1

    def __check_final_state(self):
        return all(pile == 0 for pile in self.board)

    def __generate_all_child_states(self):
        child_states: list[NimSimWorld] = []
        for i in range(self.size):
            temp_state = copy.deepcopy(self)
            for j in range(self.board[i]):
                temp_state = copy.deepcopy(temp_state)
                temp_state.pick_object_from_pile(i)
                child_states.append(temp_state)

        return child_states

    def print_board(self):
        print(self.board)
