# -*- coding: utf-8 -*-
"""
Human vs AI for Connect Four.
Input your move in the format: row,col  (e.g. 3,4)

Adapted from AlphaZero_Gomoku/human_play.py by Junxiao Song.
"""

from __future__ import print_function
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net import PolicyValueNet


class Human(object):
    """Human player."""

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location = input("Your move (row,col): ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)


def run():
    board_width = 8
    board_height = 8
    n_in_row = 4
    restricted_positions = [
        (0, 0), (0, 1), (0, 2), (0, 5), (0, 6), (0, 7),
        (1, 0), (1, 1),                    (1, 6), (1, 7),
        (2, 0),                             (2, 7),
        (5, 0),                             (5, 7),
        (6, 0), (6, 1),                    (6, 6), (6, 7),
        (7, 0), (7, 1), (7, 2), (7, 5), (7, 6), (7, 7),
    ]
    model_file = './best_policy.model'

    try:
        board = Board(width=board_width,
                      height=board_height,
                      n_in_row=n_in_row,
                      restricted_positions=restricted_positions)
        game = Game(board)

        # load the trained policy_value_net
        best_policy = PolicyValueNet(board_width, board_height,
                                     model_file=model_file)
        mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                                 c_puct=5,
                                 n_playout=400)

        # uncomment to play against pure MCTS (much weaker)
        # mcts_player = MCTS_Pure(c_puct=5, n_playout=1000)

        # human player, input your move in the format: row,col
        human = Human()

        # set start_player=0 for human first
        game.start_play(human, mcts_player, start_player=0, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
