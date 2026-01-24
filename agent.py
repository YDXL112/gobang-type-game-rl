from layers import PolicyValueNet


class Agent:
    def __init__(self):
        self.net = PolicyValueNet()

    def select_action(self, state):
        pass

    def mcts_search(self, state):
        pass

    def update(self, batch):
        pass
