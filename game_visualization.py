import kaggle_environments
from mcts import MCTS
import numpy as np
from connectfour import ConnectFour
from res_blocks import ResNet
import torch 

env = kaggle_environments.make("connectx")

class KaggleAgent:
    def __init__(self, model, game, args):
        self.model = model
        self.game = game
        self.args = args
        if self.args['search']:
            self.mcts = MCTS(self.game, self.args, self.model)

    def run(self, obs, conf):
        player = obs['mark'] if obs['mark'] == 1 else -1
        state = np.array(obs['board']).reshape(self.game.row_count, self.game.column_count)
        state[state==2] = -1

        state = self.game.change_perspective(state, player)

        if self.args['search']:
            policy = self.mcts.search(state)
        else:
            policy, _ = self.model.predict(state, augment = self.args['augment'])

        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        policy /= np.sum(policy)

        if self.args['temparature'] == 0:
            action = int(np.argmax(policy))
        elif self.args['temparature'] == float('inf'):
            action = np.random.choice([r for r in range(self.game.action_size) if policy[r] > 0])
        else:
            policy = policy ** (1 / self.args['temparature'])
            policy /= np.sum(policy)
            action = np.random.choice(self.game.action_size, p=policy)

        return action


game = ConnectFour()
player = 1

args = {
    'C': 2,
    'num_searches': 600,
    'dirichlet_epsilon': 0,
    'dirichlet_alpha': 0.3,
    'search': True,
    'temparature': 1
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(game, 9, 128, device)
model.load_state_dict(torch.load("model_7_ConnectFour.pt", map_location=device))
model.eval()


player1 = KaggleAgent(model, game, args)
player2 = KaggleAgent(model, game, args)

players = [player1.run, player2.run]

env.run(players)
html_output = env.render(mode="html")

# Write to a file
with open("connect4_game.html", "w") as f:
    f.write(html_output)