from typing import Tuple

import torch
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection

from config import EnvironmentConfig
from utils import get_num_pucks_in_area, puck_in_area


OFF_BOARD = torch.tensor([-1000.0, -1000.0])

class Environment:
    def __init__(self, environment_config: EnvironmentConfig) -> None:
        self.config = environment_config

        self.one_area, self.two_area, self.three_area, self.board_area = self._get_score_areas()

        self.puck_positions = OFF_BOARD.repeat(int(2 * self.config.num_turns)).reshape(int(2 * self.config.num_turns), 2)
        self.puck_velocities = torch.zeros(self.puck_positions.shape)
        self.turns_state = torch.tensor([self.config.num_turns, self.config.num_turns], dtype=torch.int8)
        self.current_turn = 0


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

        board_area = torch.vstack([
            one_area[0],
            three_area[1]
        ])

        return one_area, two_area, three_area, board_area


    def get_reward(self) -> float:
        score = 0
        score += 1 * get_num_pucks_in_area(self.puck_positions, *self.one_area)
        score += 2 * get_num_pucks_in_area(self.puck_positions, *self.two_area)
        score += 3 * get_num_pucks_in_area(self.puck_positions, *self.three_area)
        
        return score


    def is_finished(self) -> bool:
        return all(self.turns_state == 0.0)


    def get_state(self) -> torch.tensor:
        return torch.cat([self.puck_positions.unravel(), self.turns_state])


    def perform_action(
        self,
        x_position:torch.tensor,
        angle: torch.tensor,
        magnitude: torch.tensor
    ):
        # preprocess actions
        x_position = torch.clip(x_position, 0.0, self.config.board_width)
        angle = torch.clip(angle, 0.0, torch.pi)
        magnitude = torch.clip(magnitude, 0.0, self.config.max_agent_magnitude)

        # used for indexing the new puck
        turns_left = self.turns_state[self.current_turn]
        new_puck_index = int(self.current_turn * self.config.num_turns + turns_left - 1)

        # positions
        new_puck_position = torch.tensor([x_position, 0.0])
        self.puck_positions[new_puck_index] = new_puck_position

        # velocities
        new_puck_velocity = torch.tensor([torch.cos(angle), torch.sin(angle)]) * magnitude
        self.puck_velocities = torch.zeros(self.puck_positions.shape)
        self.puck_velocities[new_puck_index] = new_puck_velocity

        # factor by simulation_h
        simulation_fiction_coef = self.config.friction_coef ** self.config.simulation_h
        min_simulation_velocity = self.config.min_velocity * self.config.simulation_h
        max_simulation_steps = int(self.config.max_time_steps / self.config.simulation_h)

        # simulate
        self.plot()
        for simulation_step_i in range(max_simulation_steps):
            print(f"simulation_step_i: {simulation_step_i}")
            self.puck_positions += self.puck_velocities * self.config.simulation_h
            self.puck_velocities *= simulation_fiction_coef
            self.plot()

            # check for collision
            for puck_a_index, puck_a_position in enumerate(self.puck_positions):
                for a_index_offset, puck_b_position in enumerate(self.puck_positions[puck_a_index + 1:]):
                    puck_b_index = puck_a_index + a_index_offset + 1
                    if all(puck_a_position == OFF_BOARD) or all(puck_b_position == OFF_BOARD): continue

                    if torch.norm(puck_a_position - puck_b_position) < self.config.puck_radius * 2:
                        # https://en.wikipedia.org/wiki/Elastic_collision
                        total_velocity = self.puck_velocities[puck_a_index] + self.puck_velocities[puck_b_index]

                        normal_unit_vector = (puck_a_position - puck_b_position) / torch.norm(puck_a_position - puck_b_position)
                        perpendicular_unit_vector = torch.flip(normal_unit_vector, dims=(0, )) * torch.tensor([1, -1])
                        
                        normal_velocity = torch.dot(normal_unit_vector, total_velocity) * normal_unit_vector
                        perpendicular_velocity = torch.dot(perpendicular_unit_vector, total_velocity) * perpendicular_unit_vector
                        
                        # I think this is right?
                        if torch.norm(self.puck_velocities[puck_a_index]) > torch.norm(self.puck_velocities[puck_b_index]):
                            self.puck_velocities[puck_a_index] = perpendicular_velocity
                            self.puck_velocities[puck_b_index] = normal_velocity
                        else:
                            self.puck_velocities[puck_b_index] = perpendicular_velocity
                            self.puck_velocities[puck_a_index] = normal_velocity

                        # unintersect
                        
                    
            # check for out-of-bounds
            for puck_index, puck_position in enumerate(self.puck_positions):
                if not puck_in_area(puck_position, *self.board_area):
                    self.puck_positions[puck_index] = OFF_BOARD
                    self.puck_velocities[puck_index] = torch.tensor([0.0, 0.0])

            # check for low velocities
            if torch.all(torch.abs(self.puck_velocities) < min_simulation_velocity):
                break

        # end turn
        self._end_turn()

        
    def plot(self) -> str:
        # plot areas
        figure, axes = plt.subplots()
        area_rectangles = [
            plt.Rectangle(
                self.one_area[0], *(self.one_area[1] - self.one_area[0]),
                edgecolor="black",
                facecolor="grey",
                fill=True
            ),
            plt.Rectangle(
                self.two_area[0], *(self.two_area[1] - self.two_area[0]),
                edgecolor="black",
                facecolor="yellow",
                fill=True
            ),
            plt.Rectangle(
                self.three_area[0], *(self.three_area[1] - self.three_area[0]),
                edgecolor="black",
                facecolor="green",
                fill=True
            ),
        ]
        
        [axes.add_patch(patch) for patch in area_rectangles]

        # plot pucks
        puck_patches = []
        for puck_position, puck_velocity in zip(
            self.puck_positions, self.puck_velocities
        ):
            if all(puck_position == OFF_BOARD): continue

            puck_patches.append(plt.Circle(puck_position, self.config.puck_radius))
            if any(puck_velocity):
                puck_patches.append(plt.Arrow(*puck_position, *puck_velocity, color="black"))

        [axes.add_patch(patch) for patch in puck_patches]

        plt.axis("equal")
        plt.xlim([0.0, self.config.board_width])
        plt.ylim([0.0, self.three_area[1][1]])
        plt.show()


    def _end_turn(self):
        self.turns_state[self.current_turn] -= 1
        self.current_turn = 1 - self.current_turn


if __name__ == "__main__":
    environment_config = EnvironmentConfig()
    environment = Environment(environment_config=environment_config)
    environment.perform_action(
        x_position=torch.tensor(5.0),
        angle=torch.tensor(torch.pi / 2),
        magnitude=torch.tensor(0.3),
    )
    environment.perform_action(
        x_position=torch.tensor(5.5),
        angle=torch.tensor(torch.pi / 2),
        magnitude=torch.tensor(0.5),
    )
    environment.perform_action(
        x_position=torch.tensor(3.5),
        angle=torch.tensor(torch.pi / 2),
        magnitude=torch.tensor(0.7),
    )
    environment.perform_action(
        x_position=torch.tensor(5.0),
        angle=torch.tensor(torch.pi / 2),
        magnitude=torch.tensor(0.9),
    )
    environment.perform_action(
        x_position=torch.tensor(5.0),
        angle=torch.tensor(torch.pi / 2),
        magnitude=torch.tensor(0.9),
    )
