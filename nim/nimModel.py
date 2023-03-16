import logging
import os
import re
from ast import literal_eval
import numpy as np
from torch import nn, tensor, optim, load
import torch
from csv import reader
from functools import reduce
from os.path import isfile

import configs
from model import Model

logger = logging.getLogger()

class NimModel(Model):
    name = 'nim'

    def __init__(self, gamesize: int, snapshotdir: os.PathLike):
        super().__init__(gamesize, reduce(lambda x, y: x + y, range(1, gamesize+1)), snapshotdir)
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(gamesize + 1, gamesize*2)
        self.rl1 = nn.ReLU()
        self.l2 = nn.Linear(gamesize*2, gamesize*2)
        self.rl2 = nn.ReLU()
        self.l3 = nn.Linear(gamesize*2, self.classes)
        self.rl2 = nn.ReLU()
        self.sm = nn.Softmax(0)
        self.action_to_index = self.gen_action_index_dict()
        self.index_to_action = {v: k for k, v in self.action_to_index.items()}

        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)


        if isfile(f"{snapshotdir}/nim_size_{gamesize}.pth"):
            logger.info("Loading statedict")
            self.load_state_dict(load(f"{snapshotdir}/nim_size_{gamesize}.pth"))
            logger.info("Finished loading statedict")

    def gen_action_index_dict(self):
        action_to_index = dict()
        k = 0
        for i in range(self.size):
            pre = '0' * i
            post = '0' * (self.size - i - 1)
            for j in range(self.size - i):
                action_to_index[(pre + str(j+1) + post)[::-1]] = k
                k += 1
        return action_to_index

    def forward(self, x: np.ndarray):
        x = self.l1(x)
        x = self.rl1(x)
        x = self.l2(x)
        x = self.rl2(x)
        x = self.l3(x)
        x = self.rl2(x)
        return x

    def classify(self, x: np.ndarray) -> list[str]:
        x = tensor(x, dtype=torch.float)
        x = self(x)
        x = nn.Softmax(dim=0)(x)
        x = x.detach().numpy()
        return list(map(lambda x: self.index_to_action[x], np.argsort(x)))

    def load_train_data(self):
        file = reader(open('../train.csv'))
        first = True
        data = []
        for line in file:
            if first:
                first = False
                continue
            state = np.asarray(list(filter(re.compile("\d").match, line[0][1:-1])), dtype=np.int8)
            player = int(line[1])
            dist = line[2][1:-1].replace('(', '').replace(')', '').replace(' ', '').split(',')
            dist_tuples = []
            for i in range(int(len(dist)/2)):
                dist_tuples.append((dist[i*2][1:-1], float(dist[i*2+1])))
            data.append((np.append(state, [player]), {(k, v) for k, v in dist_tuples}))
        return data

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    gamesize = 4
    model = NimModel(gamesize)
    model.load_state_dict(load(f"../model_dicts/nim_size_{gamesize}.pth"))
    data = model.load_train_data()
    model.append_rbuf(data)
    model.flush_rbuf()

    torch.save(model.state_dict(), f"../model_dicts/nim_size_{gamesize}.pth")