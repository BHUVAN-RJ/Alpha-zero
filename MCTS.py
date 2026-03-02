import numpy as np
import math
import torch 
import torch.nn as nn
import torch.nn.functional as F

class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children = []

        self.visit_count = 0
        self.value_sum = 0
    
    def is_fully_expanded(self):
        return len(self.children) > 0
    
    def select(self):
        best_child = None
        #UCB = Upper Confidence Bound
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child =child
                best_ucb = ucb
        
        return best_child

    def get_ucb(self, child):
        #q == likelyhood of winninng
        #c == contant that says if we need to focus on exploration or exploitation
        if child.visit_count == 0 :
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2 # we do  1-() caz the parent and child are usually different players so we need to choose a value that makes the opponent loose
        return q_value + self.args['C'] * math.sqrt((self.visit_count) / (child.visit_count + 1)) * child.prior
    
    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player=-1)

                child = Node(self.game, self.args, child_state, self, action, prob)

                self.children.append(child)
        return child
    
    # this method is used to test the monte carlo tree search without the model but since now model is integrated commented out    
    # def simulate(self):
    #     value, is_terminal = self.game.get_value_and_terminated(self.state, self.action_taken)
    #     value = self.game.get_opponent_value(value)

    #     if is_terminal:
    #         return value
        
    #     rollout_state = self.state.copy()
    #     rollout_player = 1
    #     while True:
    #         valid_moves = self.game.get_valid_moves(rollout_state)
    #         action = np.random.choice(np.where(valid_moves == 1)[0])

    #         rollout_state = self.game.get_next_state(rollout_state, action, rollout_player)

    #         value, is_terminal = self.game.get_value_and_terminated(rollout_state, action)

    #         if is_terminal:
    #             if rollout_player == -1:
    #                 value = self.game.get_opponent_value(value)
    #             return value
            
    #         rollout_player = self.game.get_opponent(rollout_player)

    def backpropogation(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropogation(value)

class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
    
    @torch.no_grad()
    def search(self, state):
        root = Node(self.game, self.args, state, visit_count=1)

        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0)
        )

        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)
        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        policy /= np.sum(policy)

        root.expand(policy)

        for search in range(self.args['num_searches']):
            #selection
            node = root
            while node.is_fully_expanded():
                node = node.select()

            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)
            
            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)

                value = value.item()
                #expansion
                node.expand(policy)

                #simulation(random new exploration)
                
            #backprop
            node.backpropogation(value)
            
        action_probs = np.zeros(self.game.action_size)

        for child in root.children:
            action_probs[child.action_taken] = child.visit_count

        action_probs/= np.sum(action_probs)

        return action_probs

