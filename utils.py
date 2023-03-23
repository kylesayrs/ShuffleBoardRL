from typing import Tuple

import torch


def get_num_pucks_in_area(
    state: torch.tensor,
    area_bottom_left: torch.tensor,
    area_top_right: torch.tensor,
) -> int:
    return sum([
        puck_in_area(puck_position, area_bottom_left, area_top_right)
        for puck_position in state
    ])


def puck_in_area(
    puck_position: torch.tensor,
    area_bottom_left: torch.tensor,
    area_top_right: torch.tensor,
):
    position_offset = puck_position - area_bottom_left
    zero_position = torch.tensor([0.0, 0.0], device=puck_position.device)
    return torch.all(
        torch.logical_and(
            position_offset >= zero_position,
            position_offset <= (area_top_right - area_bottom_left),
        )
    )
