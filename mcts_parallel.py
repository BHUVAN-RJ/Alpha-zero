import numpy as np
import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
from mcts import Node

class MCTSParallel:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
    
    @torch.no_grad()
    def search(self, states, selfPlayGames):

        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
        )
        policy = torch.softmax(policy, axis=1).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size, size=policy.shape[0])
        
        for i, selfPlayGame in enumerate(selfPlayGames):
            spg_policy = policy[i]
            valid_moves = self.game.get_valid_moves(states[i])
            spg_policy *= valid_moves
            spg_policy /= np.sum(spg_policy)
            
            selfPlayGame.root = Node(self.game, self.args, states[i], visit_count=1)
            selfPlayGame.root.expand(spg_policy)

        for search in range(self.args['num_searches']):
            for selfPlayGame in selfPlayGames:
                selfPlayGame.node = None
                node = selfPlayGame.root
                while node.is_fully_expanded():
                    node = node.select()

                value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
                value = self.game.get_opponent_value(value)
                
                if is_terminal:
                     node.backpropogation(value)
                else:
                    selfPlayGame.node = node
            expandable_spGames = [mappingIdx for mappingIdx in range(len(selfPlayGames)) if selfPlayGames[mappingIdx].node is not None]

            if len(expandable_spGames) > 0:
                states = np.stack([selfPlayGames[mappingIdx].node.state for mappingIdx in expandable_spGames])

                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
                )
                policy = torch.softmax(policy, axis=1).cpu().numpy()
                value = value.cpu().numpy()
            
            for i, mappingIdx in enumerate(expandable_spGames):
                node = selfPlayGames[mappingIdx].node
                spg_policy, spg_value = policy[i], value[i]
                valid_moves = self.game.get_valid_moves(node.state)
                spg_policy *= valid_moves
                spg_policy /= np.sum(spg_policy)

                node.expand(spg_policy)
                node.backpropogation(spg_value)

