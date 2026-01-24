from rl_mcts import Trainer


def main():
    batch_size = 32
    lr = 1e-3
    model_save = "model.pth"
    result_save = "results.json"
    trainer = Trainer(batch_size, lr)
    pass


if __name__ == "__main__":
    main()
