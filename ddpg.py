from typing import List

import torch

from replay import Replay
from config import EnvironmentConfig


class QualityBaseModel(torch.nn.Module):
    def __init__(self, num_turns: int) -> None:
        super().__init__()
        
        self.num_turns = num_turns

        self.state_length = (self.num_turns * 4) + 3
        self.action_length = 3

        self.linear_0 = torch.nn.Linear(self.state_length + self.action_length, 512)
        self.linear_1 = torch.nn.Linear(512, 256)
        self.linear_2 = torch.nn.Linear(256, 128)
        self.linear_3 = torch.nn.Linear(128, 64)
        self.linear_4 = torch.nn.Linear(64, 1)

        self.relu = torch.nn.ReLU()


    def forward(self, state: torch.Tensor, action: torch.Tensor):
        assert len(state.shape) == 2, "QualityBaseModel forward must receive batch"
        assert len(action.shape) == 2, "QualityBaseModel forward must receive batch"
        assert state.shape[1] == self.state_length, "Invalid state length"
        assert action.shape[1] == self.action_length, "Invalid state length"

        # preprocessing
        x = torch.concat([state, action], dim=1)
        x = x.to(torch.float32)

        # network
        x = self.linear_0(x)
        x = self.relu(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_3(x)
        x = self.relu(x)
        x = self.linear_4(x)

        return x
    

class ActorBaseModel(torch.nn.Module):
    def __init__(self, num_turns: int, environment_config: EnvironmentConfig):
        super().__init__()
        
        self.num_turns = num_turns
        self.board_width = environment_config.board_width
        self.max_magnitude = environment_config.max_agent_magnitude

        self.state_length = (self.num_turns * 4) + 3
        self.action_length = 3

        self.position_bounds = [0.0, self.board_width]
        self.angle = [0.0, torch.pi]
        self.magnitude = [0.0, self.max_magnitude]

        self.linear_0 = torch.nn.Linear(self.state_length, 512)
        self.linear_1 = torch.nn.Linear(512, 256)
        self.linear_2 = torch.nn.Linear(256, 128)
        self.linear_3 = torch.nn.Linear(128, 64)
        self.linear_4 = torch.nn.Linear(64, self.action_length)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, state: torch.Tensor):
        assert len(state.shape) == 2, "ActorBaseModel forward must receive batch"
        assert state.shape[1] == self.state_length, "Invalid state length"

        # preprocessing
        x = state
        x = x.to(torch.float32)

        # network
        x = self.linear_0(x)
        x = self.relu(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_3(x)
        x = self.relu(x)
        x = self.linear_4(x)
        x = self.sigmoid(x)

        self.scale = torch.tensor([
            (self.position_bounds[1] - self.position_bounds[0]),
            (self.angle[1] - self.angle[0]),
            (self.magnitude[1] - self.magnitude[0])
        ], device=x.device, requires_grad=False)
        self.offset = torch.tensor([
            self.position_bounds[0],
            self.angle[0],
            self.magnitude[0],
        ], device=x.device, requires_grad=False)

        x = x * self.scale
        x = x + self.offset

        return x


class DDPG:
    def __init__(
        self,
        num_turns: int,
        gamma: float,
        spin_up_time: float,
        quality_lr: float,
        actor_lr: float,
        environment_config: EnvironmentConfig,
        device: str
    ) -> None:
        self.num_turns = num_turns
        self.gamma = gamma
        self.spin_up_time = spin_up_time
        self.quality_lr = quality_lr
        self.actor_lr = actor_lr
        self.e_config = environment_config
        self.device = device

        self.quality_model_query = QualityBaseModel(self.num_turns).to(self.device)
        self.quality_model_target = QualityBaseModel(self.num_turns).to(self.device)
        self.actor_model_query = ActorBaseModel(self.num_turns, self.e_config).to(self.device)
        self.actor_model_target = ActorBaseModel(self.num_turns, self.e_config).to(self.device)
        self._freeze_network(self.quality_model_target)
        self._freeze_network(self.actor_model_target)
        self.update_target_networks(1.0, 1.0)

        self.quality_optimizer, self.actor_optimizer = self._make_optimizers()

        self.quality_criterion = torch.nn.MSELoss()


    def _freeze_network(self, network: torch.nn.Module):
        for param in network.parameters():
            param.requires_grad = False

    
    def _unfreeze_network(self, network: torch.nn.Module):
        for param in network.parameters():
            param.requires_grad = True


    def _make_optimizers(self):
        quality_optimizer = torch.optim.Adam(
            self.quality_model_query.parameters(),
            lr=self.quality_lr
        )
        actor_optimizer = torch.optim.Adam(
            self.actor_model_query.parameters(),
            lr=self.actor_lr
        )

        return quality_optimizer, actor_optimizer


    def infer_action(self, state: torch.tensor, network="query"):
        with torch.no_grad():
            if network == "query":
                return self.actor_model_query(state.unsqueeze(0))[0]
            else:
                return self.actor_model_target(state.unsqueeze(0))[0]


    def update_target_networks(self, quality_momentum: float, actor_momentum: float):
        with torch.no_grad():
            for query_param, target_param in zip(
                self.quality_model_query.parameters(),
                self.quality_model_target.parameters()
            ):
                target_param.data = (
                    quality_momentum * query_param.data +
                    (1 - quality_momentum) * target_param.data
                )

            for query_param, target_param in zip(
                self.actor_model_query.parameters(),
                self.actor_model_target.parameters()
            ):
                target_param.data = (
                    actor_momentum * query_param.data +
                    (1 - actor_momentum) * target_param.data
                )


    def optimize_batch(self, batch: List[Replay], training_progress: float):
        # unpack batch
        states = torch.vstack([replay.state for replay in batch])
        actions = torch.vstack([replay.action for replay in batch])
        rewards = torch.tensor([replay.reward for replay in batch], device=self.device)
        next_states = torch.vstack([replay.next_state for replay in batch])
        is_finisheds = torch.tensor([replay.is_finished for replay in batch], device=self.device)

        # optimize networks
        quality_loss = self.optimize_quality_network(states, actions, rewards, next_states, is_finisheds)
        if training_progress > self.spin_up_time:
            actor_loss = self.optimize_actor_network(states)
        else:
            actor_loss = torch.tensor(0.0)

        return quality_loss, actor_loss


    def optimize_quality_network(
        self,
        states: torch.tensor,
        actions: torch.tensor,
        rewards: torch.tensor,
        next_states: torch.tensor,
        is_finisheds: torch.tensor
    ):
        # create target quality values
        with torch.no_grad():
            future_action_qualities = self.quality_model_target(next_states, self.actor_model_target(next_states))

            is_not_finished = 1 - is_finisheds.to(torch.float32)
            is_not_finished_transposed = torch.transpose(is_not_finished.unsqueeze(0), 0, 1)
            future_action_qualities *= is_not_finished_transposed  # zero if finished

            target_qualities = rewards# + self.gamma * torch.max(future_action_qualities, dim=1).values
            target_qualities.to(torch.float32)
            target_qualities = target_qualities.reshape((-1, 1))
        
        # forward
        self.quality_optimizer.zero_grad()
        quality_outputs = self.quality_model_query(states, actions)

        # backwards
        quality_loss = self.quality_criterion(target_qualities, quality_outputs)
        quality_loss.backward()
        self.quality_optimizer.step()

        return quality_loss
    

    def optimize_actor_network(self, states: torch.tensor):
        # forward
        self.actor_optimizer.zero_grad()
        actor_actions = self.actor_model_query(states)

        # TODO: Check whether freezing unfreezing is necessary
        self._freeze_network(self.quality_model_query)
        actor_action_qualities = self.quality_model_query(states, actor_actions)

        # backwards
        actor_loss = -1 * torch.mean(actor_action_qualities)
        actor_loss.backward()
        self.actor_optimizer.step()
        self._unfreeze_network(self.quality_model_query)

        return actor_loss
