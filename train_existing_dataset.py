import ast
import random
from _csv import reader
import re
import numpy as np

import configs
from hex.hexModel import HexModel


def train_model():
    model = HexModel(configs.size, snapshotdir=None)
    data = load_train_data()
    random.shuffle(data)
    batch_size = 25
    for i in range(0, len(data), batch_size):
        model.append_rbuf(data[i:i+batch_size])
        if i % batch_size == 0:
            model.flush_rbuf()
        if i % 50 == 0:
            model.save_model(file_name=f'{i}')

    print(data)


def load_train_data():
    file = reader(open('hex_5x5_train.csv'))
    first = True
    data = []
    for line in file:
        if first:
            first = False
            continue
        # state = np.asarray(list(filter(re.compile("\d").match, line[0][1:-1])), dtype=np.int8)
        state = line[0][1:-1]
        state = state.replace("\n", ",").replace(". ", ",").replace("'", "")\
            .replace("0.", "0").replace("  ", " ").split(",,")
        state = np.asarray([np.asarray(ast.literal_eval(x)) for x in state])

        player = int(line[1])
        dist = line[2][1:-1].replace('(', '').replace(')', '').replace(' ', '').split(',')
        dist_tuples = []
        for i in range(int(len(dist) / 2)):
            dist_tuples.append((dist[i * 2][1:-1], float(dist[i * 2 + 1])))
        data.append(((state, player), [(k, v) for k, v in dist_tuples]))
    return data


train_model()
