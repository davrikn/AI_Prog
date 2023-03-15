import re
from ast import literal_eval
import numpy as np
from torch import nn, tensor, optim, load
import torch
from csv import reader
from functools import reduce
from os.path import isfile
from model import Model

class NimModel(Model):
    def __init__(self, gamesize: int):
        self.gamesize = gamesize
        self.action_count = sigma = reduce(lambda x, y: x + y, range(1, gamesize+1))
        super().__init__()
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(gamesize + 1, gamesize*2)
        self.rl1 = nn.ReLU()
        self.l2 = nn.Linear(gamesize*2, gamesize*2)
        self.rl2 = nn.ReLU()
        self.l3 = nn.Linear(gamesize*2, sigma)
        self.rl2 = nn.ReLU()
        self.sm = nn.Softmax(0)
        self.action_to_index = self.gen_action_index_dict()
        self.index_to_action = {v: k for k, v in self.action_to_index.items()}

        if isfile(f"../model_dicts/nim_size_{gamesize}.pth"):
            self.load_state_dict(load(f"../model_dicts/nim_size_{gamesize}.pth"))

    def gen_action_index_dict(self):
        action_to_index = dict()
        k = 0
        for i in range(self.gamesize):
            pre = '0' * i
            post = '0' * (self.gamesize - i - 1)
            for j in range(self.gamesize - i):
                action_to_index[pre + str(j+1) + post] = k
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
            y_hat = np.zeros(self.action_count)
            for tup in dist_tuples:
                y_hat[self.action_to_index[tup[0][::-1]]] = tup[1]
            data.append((np.append(state, [player]), y_hat))
        return data

if __name__ == "__main__":
    gamesize = 4
    model = NimModel(gamesize)
    model.load_state_dict(load(f"../model_dicts/nim_size_{gamesize}.pth"))
    data = model.load_train_data()
    crit = nn.CrossEntropyLoss() ## TODO fix loss function, NLLLoss throws runtime error, but MSE is trash for softmax
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    g = model(tensor(data[0][0], dtype=torch.float))
    losses = []

    for i in range(len(data)):
        if i % 100 == 0:
            print(i)
        optimizer.zero_grad()
        (x, y) = data[i]
        x = tensor(x, dtype=torch.float)
        y = tensor(y, dtype=torch.float)
        out = model(x)
        loss = crit(out, y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    out = model(tensor(data[0][0], dtype=torch.float))
    torch.save(model.state_dict(), f"../model_dicts/nim_size_{gamesize}.pth")