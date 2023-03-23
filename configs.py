# Generic configs
import logging

game = 'hex'
ui = False
size = 7
simulations = 2000
log_level: int = logging.INFO


# MCTS parameters
num_episodes = 50
decay_rate = 0.75
deepflush = True
epsilon = 0.3

# ANET parameters
input_variables = 100
learning_rate = 0.05
num_hidden_layers = 50
activation_function = 'RELU'
