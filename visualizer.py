import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


def load_episode(json_path, episode_index):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    episodes = data.get("episodes", [])
    for ep in episodes:
        if int(ep.get("episode", -1)) == int(episode_index):
            return ep
    raise ValueError(f"Episode {episode_index} not found in {json_path}")


def visualize(json_path, episode_index, interval_ms=400):
    ep = load_episode(json_path, episode_index)
    moves = ep.get("moves", [])
    board = np.zeros((8, 8), dtype=np.int8)

    cmap = colors.ListedColormap(["white", "red", "blue"])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(board, cmap=cmap, norm=norm)
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_title(f"Episode {episode_index}: step 0")
    plt.grid(True, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)

    for idx, m in enumerate(moves, start=1):
        x = int(m["x"])
        y = int(m["y"])
        side = int(m["side"])
        board[y, x] = side
        im.set_data(board)
        ax.set_title(f"Episode {episode_index}: step {idx} (side={side}, move=({x},{y}))")
        plt.pause(interval_ms / 1000.0)

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize episode moves from results JSON")
    parser.add_argument("--json_path", type=str, default="results/run200.json", help="Path to results JSON")
    parser.add_argument("--episode", type=int, default=609, help="Episode index to visualize")
    parser.add_argument("--interval", type=int, default=2000, help="Interval between frames in ms")
    args = parser.parse_args()
    visualize(args.json_path, args.episode, args.interval)


if __name__ == "__main__":
    main()
