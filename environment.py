from typing import Tuple

import torch

from config import EnvironmentConfig
from utils import get_num_pucks_in_area, puck_in_area


OFF_BOARD = [-1000.0, -1000.0]

class Environment:
    def __init__(self, environment_config: EnvironmentConfig) -> None:
        self.config = environment_config

        self.one_area, self.two_area, self.three_area, self.board_area = self._get_score_areas()

        self.pucks_state = torch.tensor([OFF_BOARD] * self.config.num_turns, dtype=torch.float32)
        self.turns_state = torch.tensor([self.config.num_turns, self.config.num_turns], dtype=torch.int8)
        

    def _get_score_areas(self):
        one_area = torch.tensor([
            [0.0, 0.0],
            [self.config.board_width, self.config.one_height]
        ])

        two_area = torch.tensor([
            [0.0, one_area[1][1]],
            [self.config.board_width, one_area[1][1] + self.config.two_height]
        ])

        three_area = torch.tensor([
            [0.0, two_area[1][1]],
            [self.config.board_width, two_area[1][1] + self.config.three_height]
        ])

        board_area = three_area = torch.tensor([
            one_area[0],
            three_area[1]
        ])

        return one_area, two_area, three_area, board_area


    def get_reward(self) -> float:
        score = 0
        score += 1 * get_num_pucks_in_area(self.pucks_state, *self.one_area)
        score += 2 * get_num_pucks_in_area(self.pucks_state, *self.two_area)
        score += 3 * get_num_pucks_in_area(self.pucks_state, *self.three_area)
        
        return score


    def is_finished(self) -> bool:
        return all(self.turns_state == 0.0)


    def get_state(self) -> torch.tensor:
        return torch.cat([self.pucks_state.unravel(), self.turns_state])


    def perform_action(
        self,
        x_position:torch.tensor,
        y_position:torch.tensor,
        angle: torch.tensor,
        magnitude: torch.tensor
    ):
        # TODO: do some clipping

        # used for indexing the new puck
        turn_index, turns_left = torch.max(self.turns_state)
        new_puck_index = turn_index * self.config.num_turns + turns_left

        # positions
        new_puck_position = torch.tensor([x_position, y_position])
        self.pucks_state[new_puck_index] = new_puck_position

        # velocities
        new_puck_velocity = torch.tensor([torch.cos(angle), torch.sin(angle)]) * magnitude
        pucks_velocities = torch.zeros(self.pucks_state.shape)
        pucks_velocities[new_puck_index] = new_puck_velocity
        pucks_velocities /= self.config.simulation_h

        # simulate
        for step_i in range(self.config.simulation_h * self.config.max_time_steps):
            self.pucks_state += pucks_velocities
            pucks_velocities *= self.config.friction_coef

            # check for collision
            for puck_a_index, puck_a_position in enumerate(self.pucks_state):
                for puck_b_index, puck_b_position in enumerate(self.pucks_state):
                    if puck_a_index == puck_b_index: continue

                    if torch.norm(puck_a_position, puck_b_position) < self.config.puck_radius * 2:
                        # TODO: Something about transfer of momentum
                        #self.config.elasticity
                        pass
                    
            # check for out-of-bounds
            for puck_index, puck_position in enumerate(self.pucks_state):
                if not puck_in_area(puck_position, self.board_area):
                    self.pucks_state[puck_index] = OFF_BOARD
                    self.pucks_velocities[puck_index] = [0.0, 0.0]

            # check for low velocities
            if torch.all(pucks_velocities < self.min_velocity):
                break

        
    def __str__(self) -> str:
        pass
