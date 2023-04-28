import copy

import numpy as np
from typing import TypeVar, Callable

import configs
from game import Game
from queue import LifoQueue

HexWorld = TypeVar("HexWorld", bound="HexWorld")


def map_to_player(arr: np.ndarray) -> str:
    if arr[0] == 1:
        return "1"
    if arr[1] == 1:
        return "2"
    return "0"

class HexWorld(Game):
    def __init__(self, size: int = configs.size):
        super(HexWorld, self).__init__(size=size)
        self.size = size
        self.board = []
        self.__init_state()

    def get_children_states(self) -> list[(str, HexWorld)]:
        return [(action, self.apply(action, deepcopy=True)) for action in self.get_possible_actions()]

    def get_possible_actions(self) -> list[str]:
        actions: list[str] = []
        for i in range(self.size):
            for j in range(self.size):
                pieces = self.board[i][j]
                if pieces[0] != 0 or pieces[1] != 0:
                    continue
                actions.append(self.pad(i)+self.pad(j))
        return actions

    def apply(self, action: str, deepcopy: bool = False) -> HexWorld:
        coordinates = (int(action[:2]), int(action[-2:]))
        if coordinates[0] >= self.size or coordinates[1] >= self.size:
            raise Exception(f"Action {action} cannot be applied, as the coordinate is out of bounds")
        tmp = copy.deepcopy(self) if deepcopy else self
        if tmp.finished:
            raise Exception("Game is already finished")
        piece = tmp.board[coordinates[0]][coordinates[1]]
        if piece[0] != 0 or piece[1] != 0:
            raise Exception('Cannot place piece on an occupied coordinate')
        piece[0 if tmp.player == 1 else 1] = 1
        tmp.check_player_won(tmp.player)
        tmp.player *= -1
        return tmp

    def state_stringified(self) -> str:
        txt = "".join([map_to_player(self.board[i, j]) for j in range(self.size) for i in range(self.size)])
        txt += str(0 if self.player == 1 else 1)
        return txt

    def state(self, deNested: bool = False) -> tuple[np.ndarray, int]:
        if deNested:
            board = np.zeros((2, self.size, self.size))
            for i in range(self.size):
                for j in range(self.size):
                    board[0][i][j] = self.board[i][j][0]
                    board[1][i][j] = self.board[i][j][1]
            return board, self.player
        else:
            return self.board, self.player

    def pad(self, input: str or int, length: int = 2, start = True):
        padding = "0"*length
        out = padding+str(input) if start else str(input)+padding
        return out[-length:]

    def is_final_state(self) -> bool:
        return self.finished

    def get_utility(self) -> int:
        return self.utility

    def check_player_won(self, player: int):
        # We check if the previous player has won
        def is_player(arr: np.ndarray):
            return (arr[0] if player == 1 else arr[1]) == 1

        # Check if both sides of the players path is populated with one of its own pieces
        vectors = (self.board[0], self.board[-1]) if player == 1 else ([x[0] for x in self.board], [x[-1] for x in self.board])
        for check in [len(list(filter(is_player, vector))) != 0 for vector in vectors]:
            if not check:
                return False


        is_final_state: Callable[[tuple[int, int]], bool] = (lambda x: x[0] == self.size-1) if player == 1 else (lambda x: x[1] == self.size-1)

        frontier: LifoQueue[tuple[int, int]] = LifoQueue()
        if player == 1:
            for i in range(self.size):
                if is_player(self.board[0][i]):
                    frontier.put((0,i))
        else:
            for i in range(self.size):
                if is_player(self.board[i][0]):
                    frontier.put((i,0))
        visited = list()

        while not frontier.empty():
            next = frontier.get()
            visited.append(next)
            if is_final_state(next):
                self.finished = True
                self.utility = player
                return True
            neighbors = list(filter(lambda x: is_player(self.board[x[0]][x[1]]), self.neighbors(next)))
            neighbors = list(filter(lambda x: x not in visited, neighbors))
            for neighbor in neighbors:
                frontier.put(neighbor)
        return False


    def neighbors(self, coordinate: tuple[int,int]):
        (x,y) = coordinate
        neighbors = [(x, y - 1), (x + 1, y - 1), (x - 1, y), (x + 1, y), (x - 1, y + 1), (x, y + 1)]
        neighbors = list(filter(lambda n: n[0] >= 0 and n[0] < self.size and n[1] >= 0 and n[1] < self.size, neighbors))
        return neighbors

    def reset(self, deepcopy: bool = False, player: int = 1) -> HexWorld:
        obj = copy.deepcopy(self) if deepcopy else self
        obj.__init_state()
        return obj

    def __init_state(self, player: int = 1):
        self.board = np.zeros(shape=(self.size, self.size, 2), dtype=int)
        self.player = player
        self.utility = 0
        self.finished = False

    def __str__(self):
        hexarray = []
        for i in range(self.size):
            row = [map_to_player(self.board[i - j][j]) for j in range(i + 1)]
            # row = [str((i-j, j)) for j in range(i+1)]
            row = " ".join(row)
            hexarray.append(row)
        for i in range(1, self.size):
            row = [map_to_player(self.board[self.size - 1 - j, i + j]) for j in range(self.size - i)]
            # row = [str((self.size-1-j, i+j)) for j in range(self.size-i)]
            row = " ".join(row)
            hexarray.append(row)

        prev_len = len(hexarray[self.size - 1])
        acc = 0
        for j, i in enumerate(range(self.size - 1, -1, -1)):
            row_len = len(hexarray[i])
            diff = prev_len - row_len
            acc += int(diff / 2)
            hexarray[i] = " " * acc + hexarray[i]
            hexarray[self.size - 1 + j] = " " * acc + hexarray[self.size - 1 + j]
            prev_len = row_len
        hexarray = "\n".join(hexarray)
        return hexarray

if __name__ == "__main__":
    # No-win scenario
    #world = HexWorld(4)
    #world = HexWorld(4).apply("0000").apply("0303").apply("0100")
    #print(world)
    #print(f"Current player: {world.player}")
    #print(f"Utility: {world.get_utility()}\n")
    #
    ## P1 win scenario
    #world = HexWorld(4)
    #world = HexWorld(4).apply("0000").apply("0303").apply("0100").apply("0201").apply("0200").apply("0202").apply("0300")
    #print(world)
    #print(f"Current player: {world.player}")
    #print(f"Utility: {world.get_utility()}\n")
    #
    #
    ## P2 win scenario
    #world = HexWorld(4)
    #world = HexWorld(4).apply("0000").apply("0300").apply("0100").apply("0201").apply("0200").apply("0202").apply("0101").apply("0203")
    #print(world)
    #print(f"Current player: {world.player}")
    #print(f"Utility: {world.get_utility()}\n")
    w = HexWorld(7)
    w.board = [[
        [1,0],
        [1,0],
        [1,0],
        [0,1],
        [0,1],
        [1,0],
        [0,1]],[
        [0,1],
        [1,0],
        [1,0],
        [0,1],
        [1,0],
        [0,1],
        [1,0]],[
        [1,0],
        [1,0],
        [1,0],
        [1,0],
        [1,0],
        [1,0],
        [0,1]],[
        [1,0],
        [0,1],
        [1,0],
        [1,0],
        [1,0],
        [1,0],
        [0,1]],[
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [1,0],
        [1,0]],[
        [1,0],
        [0,1],
        [0,1],
        [0,0],
        [0,0],
        [0,1],
        [1,0]],[
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [1,0]]]
    print(w.board)
    print(w.is_final_state())
    print(w.check_player_won(1))
    print(w.check_player_won(2))

