# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Evaluation task for moving every blue cuboid into one tray."""

from __future__ import annotations

import torch
import warp as wp
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.register import register_task
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase


def _root_pos(env, asset_name: str) -> torch.Tensor:
    return wp.to_torch(env.scene[asset_name].data.root_pos_w)


def _root_lin_vel(env, asset_name: str) -> torch.Tensor:
    return wp.to_torch(env.scene[asset_name].data.root_lin_vel_w)


def _inside_tray_xy(
    object_pos: torch.Tensor,
    tray_pos: torch.Tensor,
    tray_half_extents_xy: tuple[float, float],
    object_half_extents_xy: tuple[float, float],
    margin: float,
) -> torch.Tensor:
    available_x = tray_half_extents_xy[0] - object_half_extents_xy[0] - margin
    available_y = tray_half_extents_xy[1] - object_half_extents_xy[1] - margin
    return (torch.abs(object_pos[:, 0] - tray_pos[:, 0]) <= available_x) & (
        torch.abs(object_pos[:, 1] - tray_pos[:, 1]) <= available_y
    )


def all_blue_cubes_in_tray(
    env,
    blue_cube_names: tuple[str, ...],
    tray_name: str,
    cube_sizes: tuple[tuple[float, float, float], ...],
    tray_size: tuple[float, float, float],
    xy_margin: float = 0.004,
    velocity_threshold: float = 0.12,
    max_stack_height: float = 0.16,
) -> torch.Tensor:
    """Return true when every blue cube is inside the tray and nearly stationary."""
    tray_pos = _root_pos(env, tray_name)
    tray_top = tray_pos[:, 2] + 0.5 * tray_size[2]
    result = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for name, size in zip(blue_cube_names, cube_sizes, strict=True):
        pos = _root_pos(env, name)
        inside_xy = _inside_tray_xy(
            pos,
            tray_pos,
            (0.5 * tray_size[0], 0.5 * tray_size[1]),
            (0.5 * size[0], 0.5 * size[1]),
            xy_margin,
        )
        bottom = pos[:, 2] - 0.5 * size[2]
        valid_z = (bottom >= tray_top - 0.012) & (pos[:, 2] <= tray_top + max_stack_height)
        settled = torch.linalg.vector_norm(_root_lin_vel(env, name), dim=-1) < velocity_threshold
        result &= inside_xy & valid_z & settled
    return result


def any_red_cube_in_tray(
    env,
    red_cube_names: tuple[str, ...],
    tray_name: str,
    cube_sizes: tuple[tuple[float, float, float], ...],
    tray_size: tuple[float, float, float],
) -> torch.Tensor:
    """Fail an episode when a red distractor enters the tray."""
    tray_pos = _root_pos(env, tray_name)
    tray_top = tray_pos[:, 2] + 0.5 * tray_size[2]
    result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name, size in zip(red_cube_names, cube_sizes, strict=True):
        pos = _root_pos(env, name)
        inside_xy = _inside_tray_xy(
            pos,
            tray_pos,
            (0.5 * tray_size[0], 0.5 * tray_size[1]),
            (0.5 * size[0], 0.5 * size[1]),
            0.0,
        )
        bottom = pos[:, 2] - 0.5 * size[2]
        result |= inside_xy & (bottom >= tray_top - 0.02) & (pos[:, 2] <= tray_top + 0.16)
    return result


def any_cube_dropped(env, cube_names: tuple[str, ...], minimum_height: float = -0.05) -> torch.Tensor:
    """Fail an episode if any task object falls below the tabletop scene."""
    result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in cube_names:
        result |= _root_pos(env, name)[:, 2] < minimum_height
    return result


@register_task
class BlueCubeTrayTask(TaskBase):
    """Place every blue cuboid in the tray while leaving red distractors outside."""

    def __init__(
        self,
        blue_cube_names: tuple[str, ...],
        red_cube_names: tuple[str, ...],
        cube_sizes: tuple[tuple[float, float, float], ...],
        tray_name: str,
        tray_size: tuple[float, float, float],
        episode_length_s: float = 75.0,
        task_description: str = "Pick up every blue cube and place it in the green tray. Ignore the red cubes.",
    ):
        super().__init__(episode_length_s=episode_length_s, task_description=task_description)
        self.blue_cube_names = blue_cube_names
        self.red_cube_names = red_cube_names
        self.cube_sizes = cube_sizes
        self.tray_name = tray_name
        self.tray_size = tray_size
        self.termination_cfg = TerminationsCfg(
            success=TerminationTermCfg(
                func=all_blue_cubes_in_tray,
                params={
                    "blue_cube_names": blue_cube_names,
                    "tray_name": tray_name,
                    "cube_sizes": cube_sizes[: len(blue_cube_names)],
                    "tray_size": tray_size,
                },
            ),
            red_cube_in_tray=TerminationTermCfg(
                func=any_red_cube_in_tray,
                params={
                    "red_cube_names": red_cube_names,
                    "tray_name": tray_name,
                    "cube_sizes": cube_sizes[len(blue_cube_names) :],
                    "tray_size": tray_size,
                },
            ),
            object_dropped=TerminationTermCfg(
                func=any_cube_dropped,
                params={"cube_names": (*blue_cube_names, *red_cube_names)},
            ),
        )

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self):
        return self.termination_cfg

    def get_events_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode):
        raise NotImplementedError("BlueCubeTrayTask is an evaluation task, not a MimicGen task")

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg(eye=(1.3, -1.3, 1.0), lookat=(0.35, 0.0, 0.25), origin_type="env")


@configclass
class TerminationsCfg:
    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out, time_out=True)
    success: TerminationTermCfg = MISSING
    red_cube_in_tray: TerminationTermCfg = MISSING
    object_dropped: TerminationTermCfg = MISSING
