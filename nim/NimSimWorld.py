from typing import TypeVar

import configs
from game import Game
import copy

NimSimWorld = TypeVar("NimSimWorld", bound="NimSimWorld")


class NimSimWorld(Game):
    def __init__(self, size):
        super(NimSimWorld, self).__init__(size)
        self.size = configs.size
        self.__init_board()

    def produce_initial_state(self):
        self.__init_board()

    def get_children_states(self) -> list[(str, NimSimWorld)]:
        return self.__generate_all_child_states()

    def get_child_states_enumerated(self):
        return self.__get_possible_child_states_enumerated()

    def is_final_state(self):
        return self.__check_final_state()

    def get_possible_actions(self):
        return self.__get_possible_actions()

    def enumerate_state(self) -> str:
        return ''.join(map(str, self.board))

    def get_state_utility(self) -> int:
        if self.is_final_state():
            return 1
        else:
            return 0

    def __init_board(self):
        print(self.size)
        self.board = [(i + 1) for i in range(self.size)]

    def pick_piece_from_pile(self, pile):
        if self.board[pile] <= 0:
            raise Exception("Error: Pile is already empty")
        self.board[pile] -= 1

    def __check_final_state(self) -> bool:
        print("test", self.board)
        return all(pile == 0 for pile in self.board)

    def __generate_all_child_states(self) -> list[(str, NimSimWorld)]:
        child_states: list[(str, NimSimWorld)] = []
        for i in range(self.size):
            temp_state = copy.deepcopy(self)
            for j in range(self.board[i]):
                temp_state = copy.deepcopy(temp_state)
                temp_state.pick_piece_from_pile(i)
                child_states.append((self.__get_action(i, j), temp_state))

        return child_states

    def __get_action(self, i, j) -> str:
        action = [0 for i in range(self.size)]
        action[i] = j
        return ''.join(map(str, action))

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

    def __get_possible_child_states_enumerated(self) -> list[str]:
        possible_child_states: list[str] = []
        state = self.board
        for i in range(self.size):
            for j in range(self.board[i]):
                tmp_state = copy.deepcopy(state)
                tmp_state[i] -= j + 1
                possible_child_states.append(''.join(map(str, tmp_state)))
        return possible_child_states

    def string_board_to_list(self, board_str):
        char_list = [*board_str]
        return [int(num) for num in char_list]
