import matplotlib.pyplot as plt

from config import Config
from ddpg import DDPG
#from train import train, evaluate


if __name__ == "__main__":
    config = Config(device="cpu")

    ddpg = DDPG(
        config.environment.num_turns,
        config.optimization.gamma,
        config.optimization.quality_lr,
        config.optimization.actor_lr,
        config.device
    )

    # train
    """
    ddpg, train_metrics = train(ddpg, config)

    plt.plot(train_metrics["loss"], label="loss")
    plt.legend()
    plt.show()
    

    # evaluate
    eval_metrics = evaluate(ddpg, config)
    """
