import wandb
from utils.train import train


def main():
    """
    Runs the training
    """
    wandb.login()
    train()


if __name__ == "__main__":
    main()