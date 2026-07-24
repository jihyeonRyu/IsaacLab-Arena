# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for the Franka synthetic-data evaluation environment."""

import pytest
import torch

from isaaclab_arena_environments.blue_cube_tray_environment import (
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_HORIZONTAL_APERTURE,
    CAMERA_WIDTH,
    EXTERNAL_CAMERA_FOCAL_LENGTH,
    TRAINING_TRAY_X,
    WRIST_CAMERA_FOCAL_LENGTH,
    BlueCubeTrayEnvironmentCfg,
    reset_blue_tray_layout,
)


def test_camera_intrinsics_match_franka_synthetic_generator():
    assert CAMERA_FPS == 15.0
    assert (CAMERA_WIDTH, CAMERA_HEIGHT) == (320, 256)
    assert CAMERA_HORIZONTAL_APERTURE == 20.955
    assert EXTERNAL_CAMERA_FOCAL_LENGTH == 28.0
    assert WRIST_CAMERA_FOCAL_LENGTH == 10.0


def test_environment_cfg_rejects_invalid_cube_count():
    with pytest.raises(AssertionError):
        BlueCubeTrayEnvironmentCfg(num_blue_cubes=0)


def test_environment_cfg_accepts_training_distribution_limits():
    cfg = BlueCubeTrayEnvironmentCfg(num_blue_cubes=3, num_red_cubes=2, enable_cameras=True)
    assert cfg.cube_size_range == [0.05, 0.05]
    assert cfg.tray_x == TRAINING_TRAY_X == 0.51
    assert cfg.workspace_x_bounds == [0.33, 0.70]
    assert cfg.workspace_y_bounds == [-0.34, 0.34]
    assert cfg.workspace_radius_max == 0.68
    assert cfg.tray_z == 0.013
    assert cfg.local_light_count_range == [3, 3]

def test_five_object_layout_retries_without_tray_overlap():
    class DummyAsset:
        def write_root_pose_to_sim(self, pose, env_ids=None):
            self.pose = pose.clone()

        def write_root_velocity_to_sim(self, velocity, env_ids=None):
            pass

    class DummyScene(dict):
        env_origins = torch.zeros((1, 3))

    class DummyEnv:
        num_envs = 1
        device = "cpu"
        scene = DummyScene()

    names = tuple(f"blue_cube_{index}" for index in range(3)) + tuple(
        f"red_cube_{index}" for index in range(2)
    )
    env = DummyEnv()
    for name in (*names, "green_tray"):
        env.scene[name] = DummyAsset()
    env._blue_tray_cube_sizes = {name: torch.full((1, 3), 0.065) for name in names}

    torch.manual_seed(30009)
    fixed_tray = None
    for _ in range(100):
        reset_blue_tray_layout(
            env,
            torch.tensor([0]),
            names[:3],
            names[3:],
            tuple((0.05, 0.05, 0.05) for _ in names),
            "green_tray",
            (0.22, 0.18, 0.025),
            TRAINING_TRAY_X,
            0.013,
            0.04,
            (0.33, 0.70),
            (-0.34, 0.34),
            0.68,
        )
        tray = env.scene["green_tray"].pose[0, :3]
        if fixed_tray is None:
            fixed_tray = tray.clone()
        assert torch.equal(tray, fixed_tray)
        assert float(tray[0]) == pytest.approx(TRAINING_TRAY_X)
        for name in names:
            pos = env.scene[name].pose[0, :2]
            radius = 0.5 * (2.0**0.5) * 0.065
            assert not (
                abs(float(pos[0] - tray[0])) < 0.11 + radius
                and abs(float(pos[1] - tray[1])) < 0.09 + radius
            )
