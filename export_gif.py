import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io

from connectfour import ConnectFour
from res_blocks import ResNet
from mcts import MCTS


class KaggleAgent:
    def __init__(self, model, game, args):
        self.model = model
        self.game = game
        self.args = args
        self.mcts = MCTS(self.game, self.args, self.model)

    def run(self, state, player):
        canonical = self.game.change_perspective(state, player)
        policy = self.mcts.search(canonical)
        valid_moves = self.game.get_valid_moves(canonical)
        policy *= valid_moves
        policy /= np.sum(policy)
        action = int(np.argmax(policy))
        return action


def render_board(board, move_num, player_who_moved, winner=None):
    """Render a 6x7 Connect Four board as a matplotlib figure and return PIL Image."""
    ROWS, COLS = 6, 7
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor('#1a1a8c')
    fig.patch.set_facecolor('#1a1a8c')

    for r in range(ROWS):
        for c in range(COLS):
            val = board[r, c]
            if val == 1:
                color = '#ff4444'   # Red - Player 1
            elif val == -1:
                color = '#ffff44'  # Yellow - Player 2
            else:
                color = '#111133'   # Empty (dark)
            circle = plt.Circle((c, ROWS - 1 - r), 0.42, color=color, zorder=2)
            ax.add_patch(circle)

    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(-0.5, ROWS - 0.5)
    ax.set_aspect('equal')
    ax.axis('off')

    if winner is not None:
        if winner == 0:
            title = "Draw!"
        else:
            title = f"Player {'1 (Red)' if winner == 1 else '2 (Yellow)'} Wins!"
        ax.set_title(title, color='white', fontsize=16, fontweight='bold', pad=8)
    else:
        next_player = 'Red' if player_who_moved == -1 else 'Yellow'
        ax.set_title(f"Move {move_num}  |  Next: {next_player}", color='white', fontsize=13, pad=8)

    fig.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    return img


def play_and_capture():
    game = ConnectFour()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = {
        'C': 2,
        'num_searches': 600,
        'dirichlet_epsilon': 0,
        'dirichlet_alpha': 0.3,
        'search': True,
        'temparature': 1,
    }

    model = ResNet(game, 9, 128, device)
    model.load_state_dict(torch.load("model_7_ConnectFour.pt", map_location=device))
    model.eval()

    agent = KaggleAgent(model, game, args)

    state = game.get_initial_state()
    player = 1
    frames = []

    # Capture initial board
    frames.append(render_board(state.copy(), 0, -player))

    move_num = 0
    while True:
        action = agent.run(state, player)
        state = game.get_next_state(state, action, player)
        move_num += 1

        value, is_terminal = game.get_value_and_terminated(state, action)

        if is_terminal:
            winner = player if value == 1 else 0
            frames.append(render_board(state.copy(), move_num, player, winner=winner))
            print(f"Game over in {move_num} moves. Winner: {'Player 1' if winner == 1 else 'Player 2' if winner == -1 else 'Draw'}")
            break
        else:
            frames.append(render_board(state.copy(), move_num, player))

        player = game.get_opponent(player)

    return frames


if __name__ == "__main__":
    print("Running game and capturing frames...")
    frames = play_and_capture()
    print(f"Captured {len(frames)} frames. Saving GIF...")

    output_path = "connect4_game.gif"
    # Hold first frame 1s, each move 0.8s, last frame 3s
    durations = [1000] + [800] * (len(frames) - 2) + [3000]

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=durations,
        optimize=False,
    )
    print(f"GIF saved to {output_path}")
