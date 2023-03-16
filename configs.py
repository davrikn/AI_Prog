# Generic configs
import logging

game = 'nim'
ui = False
size = 4
simulations = 50
log_level: int = logging.INFO


# MCTS parameters
num_episodes = 200
decay_rate = 0.75

# ANET parameters
input_variables = 100
learning_rate = 0.05
num_hidden_layers = 50
activation_function = 'RELU'
