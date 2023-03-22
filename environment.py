from typing import Tuple

import torch
import matplotlib.pyplot as plt
from matplotlib import animation

from config import Config
from utils import get_num_pucks_in_area, puck_in_area


OFF_BOARD = torch.tensor([-1000.0, -1000.0])

class Environment:
    def __init__(self, config: Config) -> None:
        self.e_config = config.environment

        self.one_area, self.two_area, self.three_area, self.board_area = self._get_score_areas()

        self.puck_positions = OFF_BOARD.repeat(int(2 * self.e_config.num_turns)).reshape(int(2 * self.e_config.num_turns), 2)
        self.puck_velocities = torch.zeros(self.puck_positions.shape)
        self.turns_state = torch.tensor([self.e_config.num_turns, self.e_config.num_turns], dtype=torch.int8)
        self.current_turn = 0


    def _get_score_areas(self):
        one_area = torch.tensor([
            [0.0, 0.0],
            [self.e_config.board_width, self.e_config.one_height]
        ])

        two_area = torch.tensor([
            [0.0, one_area[1][1]],
            [self.e_config.board_width, one_area[1][1] + self.e_config.two_height]
        ])

        three_area = torch.tensor([
            [0.0, two_area[1][1]],
            [self.e_config.board_width, two_area[1][1] + self.e_config.three_height]
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
        x_position = torch.clip(x_position, 0.0, self.e_config.board_width)
        angle = torch.clip(angle, 0.0, torch.pi)
        magnitude = torch.clip(magnitude, 0.0, self.e_config.max_agent_magnitude)

        # used for indexing the new puck
        turns_left = self.turns_state[self.current_turn]
        new_puck_index = int(self.current_turn * self.e_config.num_turns + turns_left - 1)

        # positions
        new_puck_position = torch.tensor([x_position, 0.0])
        self.puck_positions[new_puck_index] = new_puck_position

        # velocities
        new_puck_velocity = torch.tensor([torch.cos(angle), torch.sin(angle)]) * magnitude
        self.puck_velocities = torch.zeros(self.puck_positions.shape)
        self.puck_velocities[new_puck_index] = new_puck_velocity

        # factor by simulation_h
        simulation_fiction_coef = self.e_config.friction_coef ** self.e_config.simulation_h
        min_simulation_velocity = self.e_config.min_velocity * self.e_config.simulation_h
        max_simulation_steps = int(self.e_config.max_time_steps / self.e_config.simulation_h)

        # prepare history
        position_history = []
        velocity_history = []
        position_history.append(self.puck_positions.clone())
        velocity_history.append(self.puck_velocities.clone())

         # simulate
        for simulation_step_i in range(max_simulation_steps):
            self.puck_positions += self.puck_velocities * self.e_config.simulation_h
            self.puck_velocities *= simulation_fiction_coef

            # check for collision
            for puck_a_index, puck_a_position in enumerate(self.puck_positions):
                for a_index_offset, puck_b_position in enumerate(self.puck_positions[puck_a_index + 1:]):
                    puck_b_index = puck_a_index + a_index_offset + 1
                    if all(puck_a_position == OFF_BOARD) or all(puck_b_position == OFF_BOARD): continue

                    intersection_radius = self.e_config.puck_radius * 2 - torch.norm(puck_a_position - puck_b_position)
                    if intersection_radius > 0:
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
                        delta_a = (normal_unit_vector) * intersection_radius / 2
                        delta_b = (-1 * normal_unit_vector) * intersection_radius / 2

                        self.puck_positions[puck_a_index] += delta_a
                        self.puck_positions[puck_b_index] += delta_b

                    
            # check for out-of-bounds
            for puck_index, puck_position in enumerate(self.puck_positions):
                if not puck_in_area(puck_position, *self.board_area):
                    self.puck_positions[puck_index] = OFF_BOARD
                    self.puck_velocities[puck_index] = torch.tensor([0.0, 0.0])

            # check for low velocities
            if torch.all(torch.abs(self.puck_velocities) < min_simulation_velocity):
                break

            position_history.append(self.puck_positions.clone())
            velocity_history.append(self.puck_velocities.clone())

        self._show_animation(position_history, velocity_history, simulation_step_i)

        # end turn
        self._end_turn()


    def _show_animation(self, position_history, velocity_history, num_frames):
        figure, axes = plt.subplots()
        plt.axis("equal")
        plt.xlim([0.0, self.e_config.board_width])
        plt.ylim([0.0, self.three_area[1][1]])

        area_patches = []
        puck_patches = []
        arrow_patches = []
        animation_args = [
            position_history,
            velocity_history,
            axes,
            area_patches,
            puck_patches,
            arrow_patches,
        ]
        anim = animation.FuncAnimation(
            figure,
            self._animate,
            init_func=lambda: self._animation_init(*animation_args),
            frames=num_frames,
            fargs=animation_args,
            interval=100 * self.e_config.simulation_h,
            blit=True
        )
        plt.show()


    def _animation_init(self, position_history, velocity_history, axes, area_patches, puck_patches, arrow_patches):
        area_patches += [
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

        for puck_position, puck_velocity in zip(position_history[-1], velocity_history[-1]):
            puck_patches.append(plt.Circle(puck_position.clone(), self.e_config.puck_radius, visible=False))
            arrow_patches.append(plt.Arrow(*puck_position.clone(), *puck_velocity.clone(), color="black", visible=False))
    
        [axes.add_patch(patch) for patch in (area_patches + puck_patches + arrow_patches)]

        return area_patches + puck_patches + arrow_patches

    
    def _animate(self, frame_i, position_history, velocity_history, axes, area_patches, puck_patches, arrow_patches):
        for puck_i, (puck_position, puck_velocity) in enumerate(zip(
            position_history[frame_i], velocity_history[frame_i]
        )):
            puck_patches[puck_i].center = puck_position
            puck_patches[puck_i].set_visible(True)# = True
            arrow_patches[puck_i].remove()
            arrow_patches[puck_i] = plt.Arrow(*puck_position.clone(), *puck_velocity.clone(), color="black")

        [axes.add_patch(patch) for patch in arrow_patches]

        return area_patches + puck_patches + arrow_patches


    def _end_turn(self):
        self.turns_state[self.current_turn] -= 1
        self.current_turn = 1 - self.current_turn


if __name__ == "__main__":
    config = Config()
    environment = Environment(config)
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
