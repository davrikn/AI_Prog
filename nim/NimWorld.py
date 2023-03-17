from typing import TypeVar

import configs
from game import Game
import copy
import numpy as np

NimSimWorld = TypeVar("NimSimWorld", bound="NimSimWorld")


class NimSimWorld(Game):
    def __init__(self, size: int = configs.size, player: int = 1):
        super(NimSimWorld, self).__init__(size)
        self.size = size
        self.__init_state(player)

    def get_children_states(self) -> list[(str, Game)]:
        pass

    def get_possible_actions(self) -> list[str]:
        possible_actions: list[str] = []
        for i in range(self.size):
            for j in range(self.board[i]):
                pre = "0"*(i)
                post = "0"*(self.size-i-1)
                action = pre+str(j+1)+post
                possible_actions.append(action)
        return possible_actions

    def apply(self, action: str, deepcopy: bool = False) -> NimSimWorld:
        obj = copy.deepcopy(self) if deepcopy else self
        if len(action) != self.size:
            raise Exception(f"Action must be length {self.size}")
        if len(list(filter(lambda x: x != "0", [*action]))) != 1:
            raise Exception(f"Action must have one and only one character != 0")
        index = 0
        while(index < self.size):
            if action[index] != "0":
                break
            index += 1

        if obj.board[index] < int(action[index]):
            raise Exception(f"Draw count {action[index]} cannot be greater than the current stick count ({self.board[index]}) in the row")

        obj.board[index] -= int(action[index])
        obj.player *= 1
        return obj

    def state_stringified(self) -> str:
        return "".join([str(row) for row in self.board])+str(self.player)

    def state(self) -> tuple[list[int], int]:
        return self.board, self.player

    def is_final_state(self) -> bool:
        return len(list(filter(lambda x: x != 0, self.board))) == 0

    def get_utility(self) -> int:
        if self.is_final_state():
            return self.player
        else:
            return 0

    def reset(self, deepcopy: bool = False, player: int = 1) -> Game:
        obj = copy.deepcopy(self) if deepcopy else self
        obj.__init_state(player)
        return obj

    def __init_state(self, player: int = 1) -> None:
        self.board = [(i + 1) for i in range(self.size)]
        self.player = player


    def __str__(self):
        return ' '.join(str(i) for i in self.board)

if __name__ == "__main__":
    world = NimSimWorld(4)
    print(world)
    print(world.get_possible_actions())
    print(world.apply("1000").apply("0200").apply("0030").apply("0004"))
    print(world.is_final_state())
    print(world.player)
    print(world.get_utility())
    print(world.reset())
    print(world.state())
    print(world.state_stringified())
