from environment import BatchTracker
from agent import Agent


class Trainer:
    def __init__(self, batch_size=32, lr=1e-3):
        self.batch_size = batch_size
        self.lr = lr
        self.env = BatchTracker(batch_size)
        self.agent = Agent()

    def sample(self):
        pass

    def train(self):
        pass
