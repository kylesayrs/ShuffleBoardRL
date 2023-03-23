from abc import ABC
import torch
import numpy

from ddpg import DDPG


class Policy(ABC):
    def get_action(self, dqn, state: torch.Tensor, goal: torch.Tensor, network: str = "query") -> torch.Tensor:
        raise NotImplementedError()
    

    def update(self, training_progress: float):
        pass


class EGreedyPolicyWithNoise(Policy):
    def __init__(
        self,
        epsilon_max: float,
        epsilon_min: float,
        noise_factor: float,
        max_magnitude: float
    ):
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.noise_factor = noise_factor
        self.max_magnitude = max_magnitude

        self.epsilon = self.epsilon_max


    def get_action(self, ddpg: DDPG, state: torch.Tensor, network: str = "query") -> torch.Tensor:
        action = ddpg.infer_action(state, network=network)

        # epsilon chance of picking random (feasible) choice
        if numpy.random.choice([True, False], p=[self.epsilon, 1 - self.epsilon]):
            action = torch.rand((3, )) * torch.tensor([10.0, torch.pi, self.max_magnitude])

        else:
            action[0] += torch.normal(torch.tensor(0.0), torch.tensor(self.noise_factor * 10.0))
            action[1] += torch.normal(torch.tensor(0.0), torch.tensor(self.noise_factor * torch.pi))
            action[2] += torch.normal(torch.tensor(0.0), torch.tensor(self.noise_factor * self.max_magnitude))

            action[0] = torch.clamp(action[0], 0.0, 10.0)
            action[1] = torch.clamp(action[1], 0.0, torch.pi)
            action[2] = torch.clamp(action[2], 0.0, self.max_magnitude)

        return action.clone()
    

    def update(self, training_progress: float):
        self.epsilon = training_progress * (self.epsilon_max - self.epsilon_min) + self.epsilon_min
