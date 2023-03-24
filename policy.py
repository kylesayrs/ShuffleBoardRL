from abc import ABC
import torch
import numpy

from ddpg import DDPG


class Policy(ABC):
    def get_action(self, dqn, state: torch.Tensor, goal: torch.Tensor, network: str = "query") -> torch.Tensor:
        raise NotImplementedError()
    

    def update(self, training_progress: float):
        pass


class SpinningUpEGreedyPolicyWithNoise(Policy):
    def __init__(
        self,
        spin_up_time: float,
        epsilon_max: float,
        epsilon_min: float,
        noise_factor: float,
        board_width: float,
        max_magnitude: float,
        device: str = "cpu",
    ):
        self.spin_up_time = spin_up_time
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.noise_factor = noise_factor
        self.board_width = board_width
        self.max_magnitude = max_magnitude
        self.device = device

        self.epsilon = self.epsilon_max


    def get_action(self, ddpg: DDPG, state: torch.Tensor, network: str = "query") -> torch.Tensor:
        action = self.get_q_action(ddpg, state, network)

        # epsilon chance of picking random (feasible) choice
        if numpy.random.choice([True, False], p=[self.epsilon, 1 - self.epsilon]):
            action = (
                torch.rand((3, ), device=self.device) *
                torch.tensor([self.board_width, torch.pi, self.max_magnitude], device=self.device)
            )

        else:
            action[0] += torch.normal(torch.tensor(0.0), torch.tensor(self.noise_factor * self.board_width))
            action[1] += torch.normal(torch.tensor(0.0), torch.tensor(self.noise_factor * torch.pi))
            action[2] += torch.normal(torch.tensor(0.0), torch.tensor(self.noise_factor * self.max_magnitude))

            action[0] = torch.clamp(action[0], 0.0, self.board_width)
            action[1] = torch.clamp(action[1], 0.0, torch.pi)
            action[2] = torch.clamp(action[2], 0.0, self.max_magnitude)

        return action.clone()
    

    def get_q_action(self, ddpg: DDPG, state: torch.Tensor, network: str = "query"):
        return ddpg.infer_action(state, network=network)
    

    def update(self, training_progress: float):
        if training_progress > self.spin_up_time:
            self.epsilon = (
                ((1 - (training_progress - self.spin_up_time) / (1 - self.spin_up_time))) *
                (self.epsilon_max - self.epsilon_min) + self.epsilon_min
            )
