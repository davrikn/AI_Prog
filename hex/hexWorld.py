import copy

import numpy as np
from typing import TypeVar
from game import Game

HexWorld = TypeVar("HexWorld", bound="HexWorld")


def map_to_player(arr: np.ndarray) -> str:
    if arr[0] == 1:
        return "1"
    if arr[1] == 1:
        return "2"
    return "0"

class HexWorld(Game):
    def __init__(self, size: int):
        super(HexWorld, self).__init__(size=size)
        self.board = np.zeros(shape=(self.size, self.size, 2), dtype=int)

    def get_children_states(self) -> list[(str, HexWorld)]:
        return [(action, self.apply(action)) for action in self.get_possible_actions()]

    def get_possible_actions(self) -> list[str]:
        actions: list[str] = []
        for i in range(self.size):
            for j in range(self.size):
                pieces = self.board[i][j]
                if pieces[0] != 0 or pieces[1] != 0:
                    continue
                actions.append(self.pad(i)+self.pad(j))
        return actions

    def apply(self, action: str) -> Game:
        coordinates = (int(action[:2]), int(action[-2:]))
        if coordinates[0] >= self.size or coordinates[1] >= self.size:
            raise Exception(f"Action {action} cannot be applied, as the coordinate is out of bounds")
        tmp = copy.deepcopy(self)
        piece = tmp.board[coordinates[0]][coordinates[1]]
        if piece[0] != 0 or piece[1] != 0:
            raise Exception('Cannot place piece on an occupied coordinate')
        piece[0 if tmp.player == 1 else 1] = 1
        tmp.player *= -1
        return tmp

    def state(self, stringified: bool = False):
        if stringified:
            txt = "".join([map_to_player(self.board[i,j]) for j in range(self.size) for i in range(self.size)])
            txt += str(0 if self.player == 1 else 1)
            return txt
        else:
            return (self.board, self.player)

    def pad(self, input: str or int, length: int = 2, start = True):
        padding = "0"*length
        out = padding+str(input) if start else str(input)+padding
        return out[-length:]

    def __str__(self):
        hexarray = []
        for i in range(self.size):
            row = [map_to_player(self.board[i-j][j]) for j in range(i+1)]
            #row = [str((i-j, j)) for j in range(i+1)]
            row = " ".join(row)
            hexarray.append(row)
        for i in range(1, self.size):
            row = [map_to_player(self.board[self.size-1-j, i+j]) for j in range(self.size-i)]
            #row = [str((self.size-1-j, i+j)) for j in range(self.size-i)]
            row = " ".join(row)
            hexarray.append(row)

        prev_len = len(hexarray[self.size-1])
        acc = 0
        for j, i in enumerate(range(self.size-1, -1, -1)):
            row_len = len(hexarray[i])
            diff = prev_len-row_len
            acc += int(diff/2)
            hexarray[i] = " "*acc+hexarray[i]
            hexarray[self.size-1+j] = " "*acc+hexarray[self.size-1+j]
            prev_len = row_len
        hexarray = "\n".join(hexarray)
        return hexarray

if __name__ == "__main__":
    world = HexWorld(10)
    w2 = world.apply("0202").apply("0200").apply("0302")

    for world in w2.get_children_states():
        print(world[1])
