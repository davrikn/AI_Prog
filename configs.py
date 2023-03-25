# Generic configs
import logging

game = 'hex'
ui = False
size = 7
simulations = 25
log_level: int = logging.INFO


# MCTS parameters
num_rollouts = 250
decay_rate = 1
epsilon = 0.25
deepflush = False

# ANET parameters
input_variables = 100
learning_rate = 0.01
num_hidden_layers = 50
activation_function = 'RELU'

# NN model
model_dir = 'models'
