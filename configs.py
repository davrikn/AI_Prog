# Generic configs
import logging

import torch
import torch.nn as nn


game = 'hex'
display_UI = False
size = 7
OHT_size = 7
simulations = 25
log_level: int = logging.INFO
save_data = False
M = 41
G = 250


# MCTS parameters
num_episodes = 1000
num_rollouts = 500
decay_rate = 1
epsilon = 0.9
epsilon_decay = 0.99
deepflush = False

# ANET parameters
structure = [[128, 'relu']]
loss_function = nn.MSELoss()  # SGDLoss
optimizer = "adam"  # adagrad, sgd, rmsprop or adam
input_variables = 200
learning_rate = 0.001
num_hidden_layers = 50
epochs = 75

# NN model
model_dir = 'models'
