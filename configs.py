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
save_data = True


# MCTS parameters
num_episodes = 200
num_rollouts = 500
decay_rate = 1
epsilon = 1
deepflush = False

# ANET parameters
loss_function = nn.MSELoss()  # SGDLoss
optimizer = torch.optim.Adam  # SGD
input_variables = 100
learning_rate = 0.01
num_hidden_layers = 50
activation_function = 'RELU'

# NN model
model_dir = 'models'
