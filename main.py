import numpy
import matplotlib.pyplot as plt

from config import Config
from ddpg import DDPG

from replay import CircularBuffer, Replay
from environment import ShuffleBoardEnvironment
from policy import Policy, SpinningUpEGreedyPolicyWithNoise
#from train import train, evaluate


def train(ddpg: DDPG, policy: Policy, config: Config):
    replay_buffer = CircularBuffer(config.optim.replay_buffer_size)
    quality_losses = []
    actor_losses = []
    metrics = {}
    zero_rewards = []
    one_rewards = []
    metrics["zero_rewards"] = []
    metrics["one_rewards"] = []

    for episode_i in range(config.optim.num_episodes):
        training_progress = episode_i / config.optim.num_episodes  # used later

        environment = ShuffleBoardEnvironment(config.env, config.device)
        while not environment.is_finished():
            # do action in environment
            state = environment.get_state()
            action = policy.get_action(ddpg, state, "query")
            environment.perform_action(action)
            reward = environment.get_reward()
            if environment.current_turn == 0:
                zero_rewards.append(environment.get_reward())
            else:
                one_rewards.append(environment.get_reward())

            environment.end_turn()
            next_state = environment.get_state()
            is_finished = environment.is_finished()

            # save replay of action
            replay = Replay(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                is_finished=is_finished,
            )
            replay_buffer.enqueue(replay)

        # cycle: perform optimization
        if episode_i % config.optim.episodes_per_cycle == 0:

            cum_quality_loss, cum_actor_loss = 0.0, 0.0
            for _ in range(config.optim.batches_per_cycle):
                batch = replay_buffer.get_random_batch(config.optim.batch_size)
                quality_loss, actor_loss = ddpg.optimize_batch(batch, training_progress)

                cum_quality_loss += quality_loss.item()
                cum_actor_loss += actor_loss.item()

            # update ddpg targets and policy
            ddpg.update_target_networks(config.optim.quality_momentum, config.optim.actor_momentum)
            policy.update(training_progress)
            quality_losses.append(cum_quality_loss / config.optim.batches_per_cycle)
            actor_losses.append(cum_actor_loss / config.optim.batches_per_cycle)

        # logging
        if episode_i % config.logging_rate == 0:
            #visualize_game(ddpg, config, num_turns=1)
            tmp_environment = ShuffleBoardEnvironment(config.env, config.device)
            tmp_action = policy.get_q_action(ddpg, tmp_environment.get_state(), "query")
            print(tmp_action)
            environment.perform_action(tmp_action)
            tmp_environment.end_turn()
            print(policy.get_q_action(ddpg, tmp_environment.get_state(), "query"))
            if config.verbosity >= 1:
                print(
                    f" | {episode_i} / {config.optim.num_episodes}"
                    f" | {numpy.mean(zero_rewards):.1f} : {numpy.mean(one_rewards):.1f}"
                , end="")

            if config.verbosity >= 2:
                q_loss = quality_losses[-1] if len(quality_losses) > 0 else 0.0
                a_loss = actor_losses[-1] if len(actor_losses) > 0 else 0.0
                print(
                    f" | q_loss: {q_loss:.3f}"
                    f" | a_loss: {a_loss:.3f}"
                    f" | epsilon: {policy.epsilon:.2f}"
                , end="")

            if config.verbosity > 0:
                print()

            metrics["zero_rewards"].append(numpy.mean(zero_rewards))

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
        environment.perform_action(action, animate=True)
        environment.end_turn()
        num_turns_played += 1



if __name__ == "__main__":
    config = Config(device="cpu")

    ddpg = DDPG(
        config.env.num_turns,
        config.optim.gamma,
        config.optim.spin_up_time,
        config.optim.quality_lr,
        config.optim.actor_lr,
        config.env,
        config.device
    )

    # train
    ddpg, train_metrics = train(
        ddpg,
        SpinningUpEGreedyPolicyWithNoise(
            config.optim.spin_up_time,
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
