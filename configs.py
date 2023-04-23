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
M = 51
G = 500


# MCTS parameters
num_episodes = 500
num_rollouts = 500
decay_rate = 0.95
epsilon = 0.5
epsilon_decay = 0.985
deepflush = False

# ANET parameters
structure = [[128, 'relu']]
loss_function = nn.MSELoss()  # SGDLoss
optimizer = "adam"  # adagrad, sgd, rmsprop or adam
input_variables = 200
learning_rate = 0.001
num_hidden_layers = 50
epochs = 50

# NN model
model_dir = 'models'
