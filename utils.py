import csv
from os.path import exists

import configs


def save_train_data(buffer):
    if not exists(f'hex_{configs.size}x{configs.size}_train.csv'):
        with open(f'hex_{configs.size}x{configs.size}_train.csv', 'w', newline='') as f:
            writer = csv.writer(f)

            fields = ["state", "player", "mcts_distribution"]
            writer.writerow(fields)

            f.close()

    with open(f'hex_{configs.size}x{configs.size}_train.csv', 'a', newline='') as f:
        writer = csv.writer(f)

        for data in buffer:
            state = data[0][0]
            player = data[0][1]
            dists = data[1]

            writer.writerow([state, player, dists])

    f.close()