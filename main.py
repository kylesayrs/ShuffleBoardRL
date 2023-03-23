import numpy
import matplotlib.pyplot as plt

from config import Config
from ddpg import DDPG

from replay import CircularBuffer, Replay
from environment import ShuffleBoardEnvironment
from policy import Policy, EGreedyPolicyWithNoise
#from train import train, evaluate


def train(ddpg: DDPG, policy: Policy, config: Config):
    replay_buffer = CircularBuffer(config.optim.replay_buffer_size)
    quality_losses = []
    actor_losses = []
    metrics = {}
    zero_rewards = []
    one_rewards = []

    for episode_i in range(config.optim.num_episodes):
        environment = ShuffleBoardEnvironment(config.env, config.device)

        while not environment.is_finished():
            # do action in environment
            state = environment.get_state()
            action = policy.get_action(ddpg, state, "query")
            environment.perform_action(action)

            # save replay of action
            replay = Replay(
                state=state,
                action=action,
                reward=environment.get_reward(),
                next_state=environment.get_state(),
                is_finished=True#environment.is_finished()
            )
            replay_buffer.enqueue(replay)

            # end turn
            policy.update(episode_i / config.optim.num_episodes)
            zero_rewards.append(environment.get_reward(0))
            one_rewards.append(environment.get_reward(1))
            break
            environment.end_turn()

        # cycle: perform optimization
        if episode_i % config.optim.episodes_per_cycle == 0:

            cum_quality_loss, cum_actor_loss = 0.0, 0.0
            for _ in range(config.optim.batches_per_cycle):
                batch = replay_buffer.get_random_batch(config.optim.batch_size)
                quality_loss, actor_loss = ddpg.optimize_batch(batch)

                cum_quality_loss += quality_loss.item()
                cum_actor_loss += actor_loss.item()

            ddpg.update_target_networks(config.optim.quality_momentum, config.optim.actor_momentum)
            quality_losses.append(cum_quality_loss / config.optim.batches_per_cycle)
            actor_losses.append(cum_actor_loss / config.optim.batches_per_cycle)

        # logging:
        if episode_i % config.logging_rate == 0:
            #visualize_game(ddpg, config, num_turns=1)
            print(policy.get_q_action(ddpg, ShuffleBoardEnvironment(config.env, config.device).get_state(), "query"))
            if config.verbosity >= 1:
                print(
                    f" | {episode_i} / {config.optim.num_episodes}"
                    f" | {numpy.mean(zero_rewards):.1f} : {numpy.mean(one_rewards):.1f}"
                , end="")

            if config.verbosity >= 2:
                q_loss = quality_losses[-1] if len(quality_losses) > 0 else 0.0
                a_loss = actor_losses[-1] if len(actor_losses) > 0 else 0.0
                print(
                    f" | q_loss: {q_loss:.6f}"
                    f" | a_loss: {a_loss:.6f}"
                    f" | epsilon: {policy.epsilon:.2f}"
                , end="")

            if config.verbosity > 0:
                print()

            zero_rewards = []
            one_rewards = []

    return ddpg, metrics


def visualize_game(ddpg: DDPG, config: Config, num_turns: int):
    environment = ShuffleBoardEnvironment(config.env)

    num_turns_played = 0
    while not environment.is_finished() and num_turns_played < num_turns:
        # do action in environment
        state = environment.get_state()
        action = ddpg.infer_action(state, "query")
        print(action)
        environment.perform_action(action, animate=True)
        environment.end_turn()
        num_turns_played += 1



if __name__ == "__main__":
    config = Config(device="cpu")

    ddpg = DDPG(
        config.env.num_turns,
        config.optim.gamma,
        config.optim.quality_lr,
        config.optim.actor_lr,
        config.env,
        config.device
    )

    # train
    ddpg, train_metrics = train(
        ddpg,
        EGreedyPolicyWithNoise(
            config.optim.epsilon_max,
            config.optim.epsilon_min,
            config.optim.noise_factor,
            config.env.max_agent_magnitude
        ),
        config
    )

    visualize_game(ddpg, config, num_turns=10)

    """
    ddpg, train_metrics = train(ddpg, config)

    plt.plot(train_metrics["loss"], label="loss")
    plt.legend()
    plt.show()
    

    # evaluate
    eval_metrics = evaluate(ddpg, config)
    """
