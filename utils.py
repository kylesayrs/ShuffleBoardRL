from typing import Tuple

import torch


def get_num_pucks_in_area(
    state: torch.tensor,
    area_bottom_left: torch.tensor,
    area_top_right: torch.tensor,
) -> int:
    positions_offset = state - area_bottom_left
    zeros_position = torch.tensor([0.0, 0.0])

    return torch.sum(
        torch.all(
            torch.logical_and(
                positions_offset >= zeros_position,
                positions_offset <= area_top_right,
            ), axis=1
        )
    )


def puck_in_area(
    puck_position: torch.tensor,
    area_bottom_left: torch.tensor,
    area_top_right: torch.tensor,
):
    position_offset = puck_position - area_bottom_left
    zero_position = torch.tensor([0.0, 0.0])
    return torch.all(
        torch.logical_and(
            position_offset >= zero_position,
            position_offset <= area_top_right,
        )
    )


if __name__ == "__main__":
    state = torch.tensor([
        [0.5, 0.5],
        [-0.5, -0.5],
        [0.5, -0.5],
        [-0.5, 0.5],
        [1.5, 0.5],
        [0.5, 1.5],
        [1.5, 1.5],
    ])

    area = torch.tensor([
        [0.0, 0.0],
        [1.0, 1.0]
    ])

    print(get_num_pucks_in_area(state, *area))
