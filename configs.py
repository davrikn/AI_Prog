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


# MCTS parameters
num_episodes = 1000
num_rollouts = 200
decay_rate = 1
epsilon = 0.25
deepflush = False

# ANET parameters
loss_function = nn.MSELoss()  # SGDLoss
optimizer = "adam"  # adagrad, sgd, rmsprop or adam
input_variables = 200
learning_rate = 0.0001
num_hidden_layers = 50
activation_function = 'RELU'

# NN model
model_dir = 'models'
