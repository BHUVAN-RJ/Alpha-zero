from tictactoe import TicTacToe
from connectfour import ConnectFour
from alpha_zero import AlphaZero
from alpha_zero_parallel import AlphaZeroParallel
import numpy as np
import res_blocks as rb
import torch 


game = ConnectFour()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = {
    'C': 2,
    'num_searches': 600,
    'num_iterations': 8,
    'num_selfPlay_iterations': 500,
    'num_parallel_games': 100,
    'num_epochs': 4,
    'batch_size': 128,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
}

#tictacto parameters
# args = {
#     'C': 2,
#     'num_searches': 60,
#     'num_iterations': 3,
#     'num_selfPlay_iterations': 500,
#     'num_epochs': 4,
#     'batch_size': 64,
#     'temperature': 1.25,
#     'dirichlet_epsilon': 0.25,
#     'dirichlet_alpha': 0.3
# }


model = rb.ResNet(game, 9, 128, device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

alphaZero_model = AlphaZeroParallel(model, optimizer, game, args)
alphaZero_model.learn()
