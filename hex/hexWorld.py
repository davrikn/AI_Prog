import copy

import numpy as np
from typing import TypeVar, Callable
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

    def apply(self, action: str, deepcopy: bool = False) -> HexWorld:
        coordinates = (int(action[:2]), int(action[-2:]))
        if coordinates[0] >= self.size or coordinates[1] >= self.size:
            raise Exception(f"Action {action} cannot be applied, as the coordinate is out of bounds")
        tmp = copy.deepcopy(self) if deepcopy else self
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

    def is_final_state(self):
        return self.check_player_won(self.player*-1)

    def utility(self):
        if self.check_player_won(1):
            return 1
        elif self.check_player_won(-1):
            return -1
        else:
            return 0

    def check_player_won(self, player: int):
        # We check if the previous player has won
        def is_player(arr: np.ndarray):
            return (arr[0] if player == 1 else arr[1]) == 1

        # Check if both sides of the players path is populated with one of its own pieces
        vectors = (self.board[0], self.board[-1]) if player == 1 else ([x[0] for x in self.board], [x[-1] for x in self.board])
        for check in [len(list(filter(is_player, vector))) != 0 for vector in vectors]:
            if not check:
                return False

        # TODO DFS search to find path from one side to the other

        is_final_state: Callable[[tuple[int, int]], bool] = (lambda x: x[0] == 3) if player == 1 else (lambda x: x[1] == 3)

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


    # No-win scenario
    world = HexWorld(4)
    world = HexWorld(4).apply("0000").apply("0303").apply("0100")
    print(world)
    print(f"Current player: {world.player}")
    print(f"Utility: {world.utility()}\n")


    # P1 win scenario
    world = HexWorld(4)
    world = HexWorld(4).apply("0000").apply("0303").apply("0100").apply("0201").apply("0200").apply("0202").apply("0300")
    print(world)
    print(f"Current player: {world.player}")
    print(f"Player 1 won: {world.is_final_state()}")
    print(f"Utility: {world.utility()}\n")


    # P2 win scenario
    world = HexWorld(4)
    world = HexWorld(4).apply("0000").apply("0300").apply("0100").apply("0201").apply("0200").apply("0202").apply("0101").apply("0203")
    print(world)
    print(f"Current player: {world.player}")
    print(f"Player 2 won: {world.is_final_state()}")
    print(f"Utility: {world.utility()}\n")

