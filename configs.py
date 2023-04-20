# Generic configs
import logging

import torch
import torch.nn as nn


game = 'hex'
display_UI = False
size = 4
OHT_size = 7
simulations = 25
log_level: int = logging.INFO
save_data = False
M = 21
G = 250


# MCTS parameters
num_episodes = 200
num_rollouts = 300
decay_rate = 1
epsilon = 0.25
deepflush = False

# ANET parameters
structure = [[128, 'relu'], [256, 'relu']]
loss_function = nn.MSELoss()  # SGDLoss
optimizer = "adam"  # adagrad, sgd, rmsprop or adam
input_variables = 200
learning_rate = 0.001
num_hidden_layers = 50
epochs = 25

# NN model
model_dir = 'models'
