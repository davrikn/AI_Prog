from game import Game
import copy


class NimSimWorld(Game):
    def __init__(self, size: int = 4):
        super(NimSimWorld, self).__init__(size)
        self.board = self.__init_board()

    def produce_initial_state(self):
        pass

    def get_children_states(self):
        return self.__generate_all_child_states()

    def is_final_state(self):
        return self.__check_final_state()

    def get_possible_actions(self):
        return self.__get_possible_actions()

    def __init_board(self):
        board = []
        for i in range(self.size):
            board.append(i + 1)
        return board

    def pick_piece_from_pile(self, pile):
        if self.board[pile] <= 0:
            raise Exception("Error: Pile is already empty")
        self.board[pile] -= 1

    def __check_final_state(self) -> bool:
        return all(pile == 0 for pile in self.board)

    def __generate_all_child_states(self):
        child_states: list[NimSimWorld] = []
        for i in range(self.size):
            temp_state = copy.deepcopy(self)
            for j in range(self.board[i]):
                temp_state = copy.deepcopy(temp_state)
                temp_state.pick_piece_from_pile(i)
                child_states.append(temp_state)

        return child_states

    def __get_possible_actions(self) -> list[str]:
        possible_actions: list[str] = []
        action = [0 for i in range(self.size)]
        for i in range(self.size):
            for j in range(self.board[i]):
                if i > 0 and j == 0:
                    continue
                tmp_action = copy.deepcopy(action)
                tmp_action[i] = j
                possible_actions.append(''.join(map(str, tmp_action)))
                # possible_actions.append(tmp_action)
        return possible_actions

    def print_board(self):
        print(self.board)
