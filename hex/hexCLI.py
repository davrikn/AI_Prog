import time

import configs
from hex.hexModel import HexModel
from hex.hexWorld import HexWorld
from os import system

machine_move_delay = 3

if __name__ == "__main__":
    game = HexWorld(configs.size)
    model = HexModel(configs.size, '../model_dicts')

    grid = []
    for i in range(configs.size):
        row = []
        for j in range(i+1):
            row.append((i-j, j))
        grid.append(row)
    for i in range(1, configs.size):
        row = []
        for j in range(configs.size-i):
            row.append((configs.size-1-j, i+j))
        grid.append(row)

    def get_machine_play():
        time.sleep(machine_move_delay)
        actions = model.classify(game.state(deNested=True))
        for action in actions:
            try:
                game.apply(action)
                print(f"Machine played {action}")
                break
            except:
                pass

    def get_human_play():
            row = int(input("What row?"))-1
            col = int(input("What column?"))-1
            coordinates = grid[row][col]
            action = ("00"+str(coordinates[0]))[-2:] + ("00"+str(coordinates[1]))[-2:]
            try:
                game.apply(action)
            except:
                print(f"Illegal action {action}")

    p1 = get_machine_play
    p2 = get_machine_play

    while not game.finished:
        print(f"\n{game}")
        if game.player == 1:
            p1()
        else:
            p2()


    print(f"{game}\n Player {1 if game.get_utility() == 1 else 2} won")