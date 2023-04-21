import ast
import random
from _csv import reader
import re
import numpy as np

import configs
from hex.hexModel import HexModel

hex_5x5_path = 'hex_5x5_train.csv'
hex_4x4_path = 'hex_4x4_train.csv'


def train_model():
    model = HexModel(configs.size, snapshotdir=None)
    data = load_train_data(hex_4x4_path)
    random.shuffle(data)
    batch_size = 20
    for i in range(0, len(data), batch_size):
        model.append_rbuf(data[i:i+batch_size])
        if i % batch_size == 0:
            model.flush_rbuf()
            print(f'Trained on {i}/{len(data)} data..')

        if i % 100 == 0:
            model.save_model(file_name=f'{i}')
            # for param in model.parameters():
            #     print(param.data)


def load_train_data(file_path):
    file = reader(open(file_path))
    first = True
    data = []
    for line in file:
        if first:
            first = False
            continue
        # state = np.asarray(list(filter(re.compile("\d").match, line[0][1:-1])), dtype=np.int8)
        player = int(line[1])
        if player == 1:
            pass
            # continue
        state = line[0][1:-1]
        state = state.replace("\n", ",").replace(". ", ",").replace("'", "")\
            .replace("0.", "0").replace("  ", " ").split(",,")
        state = np.asarray([np.asarray(ast.literal_eval(x)) for x in state])

        dist = line[2][1:-1].replace('(', '').replace(')', '').replace(' ', '').split(',')
        dist_tuples = []
        for i in range(int(len(dist) / 2)):
            dist_tuples.append((dist[i * 2][1:-1], float(dist[i * 2 + 1])))
        data.append(((state, player), [(k, v) for k, v in dist_tuples]))
    return data


train_model()
