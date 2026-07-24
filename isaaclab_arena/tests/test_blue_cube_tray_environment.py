# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for the Franka synthetic-data evaluation environment."""

import pytest

from isaaclab_arena_environments.blue_cube_tray_environment import (
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_HORIZONTAL_APERTURE,
    CAMERA_WIDTH,
    EXTERNAL_CAMERA_FOCAL_LENGTH,
    WRIST_CAMERA_FOCAL_LENGTH,
    BlueCubeTrayEnvironmentCfg,
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
    assert cfg.cube_size_range == [0.05, 0.065]
    assert cfg.workspace_x_bounds == [0.33, 0.70]
    assert cfg.workspace_y_bounds == [-0.34, 0.34]
    assert cfg.workspace_radius_max == 0.68
    assert cfg.tray_z == 0.013
    assert cfg.local_light_count_range == [3, 3]
