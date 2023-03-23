# Generic configs
import logging

game = 'hex'
ui = False
size = 4
simulations = 25
log_level: int = logging.INFO


# MCTS parameters
num_episodes = 100
decay_rate = 0.75
epsilon = 0.3
deepflush = False

# ANET parameters
input_variables = 100
learning_rate = 0.05
num_hidden_layers = 50
activation_function = 'RELU'

# NN model
model_dir = 'models'
