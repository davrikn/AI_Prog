# Generic configs
import logging

game = 'hex'
ui = False
size = 7
simulations = 25
log_level: int = logging.INFO


# MCTS parameters
num_episodes = 25
decay_rate = 1
epsilon = 0.3
deepflush = False

# ANET parameters
input_variables = 100
learning_rate = 0.05
num_hidden_layers = 50
activation_function = 'RELU'

# NN model
model_dir = 'models'
