from typing import List

import torch

from replay import Replay


class QualityBaseModel(torch.nn.Module):
    def __init__(self, num_turns: int) -> None:
        super().__init__()
        
        self.num_turns = num_turns

        self.state_length = (self.num_turns * 2) + 2
        self.action_length = 3

        self.linear_0 = torch.nn.Linear(self.state_length + self.action_length, self.state_length)
        self.linear_1 = torch.nn.Linear(self.state_length, self.state_length)
        self.linear_2 = torch.nn.Linear(self.state_length, 1)

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

        return x
    

class ActorBaseModel(torch.nn.Module):
    def __init__(self, num_turns: int) -> None:
        super().__init__()
        
        self.num_turns = num_turns

        self.state_length = (self.num_turns * 2) + 2
        self.action_length = 3

        self.linear_0 = torch.nn.Linear(self.state_length + self.action_length, self.state_length)
        self.linear_1 = torch.nn.Linear(self.state_length, self.state_length)
        self.linear_2 = torch.nn.Linear(self.state_length, self.action_length)

        self.relu = torch.nn.ReLU()


    def forward(self, state: torch.Tensor, action: torch.Tensor):
        assert len(state.shape) == 2, "ActorBaseModel forward must receive batch"
        assert len(action.shape) == 2, "ActorBaseModel forward must receive batch"
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

        return x


class DDPG:
    def __init__(
        self,
        num_turns: int,
        gamma: float,
        quality_lr: float,
        actor_lr: float,
        device: str
    ) -> None:
        self.num_turns = num_turns
        self.gamma = gamma
        self.quality_lr = quality_lr
        self.actor_lr = actor_lr
        self.device = device

        self.quality_model_query = QualityBaseModel(self.num_turns).to(self.device)
        self.quality_model_target = QualityBaseModel(self.num_turns).to(self.device)
        self.actor_model_query = ActorBaseModel(self.num_turns).to(self.device)
        self.actor_model_target = ActorBaseModel(self.num_turns).to(self.device)
        self.update_target_networks(1.0)

        self.quality_optimizer, self.actor_optimizer = self._make_optimizers()

        self.quality_criterion = torch.nn.MSELoss()
        self.actor_criterion = torch.nn.L1Loss()


    def _make_optimizers(self):
        quality_optimizer = torch.optim.Adam(
            self.quality_model_query.parameters(),
            lr=self.quality_lr
        )
        actor_optimizer = torch.optim.Adam(
            self.actor_model_query.parameters(),
            lr=self.quality_lr
        )

        return quality_optimizer, actor_optimizer


    def infer_action():
        pass


    def update_target_networks(self, momentum: float):
        with torch.no_grad():
            for query_param, target_param in zip(
                self.quality_model_query.parameters(),
                self.quality_model_target.parameters()
            ):
                target_param.data = (
                    momentum * query_param.data +
                    (1 - momentum) * target_param.data
                )

            for query_param, target_param in zip(
                self.actor_model_query.parameters(),
                self.actor_model_target.parameters()
            ):
                target_param.data = (
                    momentum * query_param.data +
                    (1 - momentum) * target_param.data
                )


    def optimize_batch(self, batch: List[Replay]):
        # unpack batch
        states = torch.vstack([replay.state for replay in batch])
        actions = torch.vstack([replay.action for replay in batch])
        rewards = torch.tensor([replay.reward for replay in batch], device=self.device)
        next_states = torch.vstack([replay.next_state for replay in batch])
        is_finisheds = torch.tensor([replay.is_finished for replay in batch], device=self.device)

        # optimize networks
        quality_loss = self.optimize_quality_network()
        actor_loss = self.optimize_actor_network()

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

            target_qualities = rewards + self.gamma * torch.max(future_action_qualities, dim=1).values
            target_qualities.to(torch.float32)
        
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
        with torch.no_grad():
            actor_action_qualities = self.quality_model_query(states, actor_actions)

        # backwards
        zeros = torch.zeros(actor_action_qualities.shape)
        actor_loss = self.actor_criterion(zeros, -1 * actor_action_qualities)
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss
