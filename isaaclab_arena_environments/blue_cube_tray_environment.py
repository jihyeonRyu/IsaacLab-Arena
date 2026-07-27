# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Arena environment matching the Franka blue-cuboid-to-green-tray dataset."""

from __future__ import annotations

import math
import torch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


BLUE = (0.0426, 0.1420, 0.9345)
RED = (0.9899, 0.0162, 0.0290)
GREEN = (0.1244, 0.5804, 0.1832)
TABLETOP_FRANKA_JOINT_POS = {
    "panda_joint1": 0.50,
    "panda_joint2": -0.569,
    "panda_joint3": 0.0,
    "panda_joint4": -2.810,
    "panda_joint5": 0.0,
    "panda_joint6": 3.037,
    "panda_joint7": 0.741,
    "panda_finger_joint.*": 0.040,
}

# Same closed-room geometry used by the vectorized synthetic-data generator.
# The Arena default env spacing is 30 m, so these 3.9 m rooms cannot overlap.
ROOM_FACES = (
    ("background_back", (3.9, 0.02, 4.0), (0.0, 1.94, 0.95)),
    ("background_front", (3.9, 0.02, 4.0), (0.0, -1.94, 0.95)),
    ("background_left", (0.02, 3.9, 4.0), (-1.94, 0.0, 0.95)),
    ("background_right", (0.02, 3.9, 4.0), (1.94, 0.0, 0.95)),
    ("background_floor", (3.9, 3.9, 0.02), (0.0, 0.0, -1.059)),
    ("background_ceiling", (3.9, 3.9, 0.02), (0.0, 0.0, 2.96)),
)
BACKGROUND_PALETTE = (
    (0.04, 0.06, 0.10),
    (0.12, 0.20, 0.36),
    (0.38, 0.24, 0.12),
    (0.18, 0.30, 0.22),
    (0.30, 0.18, 0.38),
    (0.32, 0.33, 0.36),
    (0.48, 0.40, 0.28),
    (0.10, 0.32, 0.34),
)
LIGHT_PALETTE = ((1.0, 0.92, 0.80), (0.82, 0.90, 1.0), (1.0, 0.98, 0.92))
TRAY_PALETTE = ((0.10, 0.55, 0.22), (0.95, 0.78, 0.18), (0.28, 0.28, 0.30))
TABLE_PALETTE = ((0.55, 0.47, 0.37), (0.72, 0.72, 0.68), (0.38, 0.42, 0.44))
CAMERA_FPS = 15.0
CAMERA_HEIGHT = 256
CAMERA_WIDTH = 320
CAMERA_HORIZONTAL_APERTURE = 20.955
EXTERNAL_CAMERA_FOCAL_LENGTH = 28.0
WRIST_CAMERA_FOCAL_LENGTH = 10.0
WORKSPACE_X_BOUNDS = (0.33, 0.70)
WORKSPACE_Y_BOUNDS = (-0.34, 0.34)
WORKSPACE_RADIUS_MAX = 0.68
TRAINING_TRAY_X = 0.51


def update_blue_tray_wrist_camera(
    env,
    env_ids: torch.Tensor | None,
    camera_name: str,
    ee_frame_name: str,
    eye_offset: tuple[float, float, float],
    target_offset: tuple[float, float, float],
) -> None:
    """Match the generator's 15 Hz world-up look-at wrist camera update."""
    from isaaclab.utils.math import quat_apply

    ee_frame = env.scene[ee_frame_name]
    indices = slice(None) if env_ids is None else env_ids
    ee_pos_w = ee_frame.data.target_pos_w[indices, 0]
    ee_quat_w = ee_frame.data.target_quat_w[indices, 0]
    count = int(ee_pos_w.shape[0])
    eye_local = torch.tensor(eye_offset, device=env.device, dtype=torch.float32).expand(count, -1)
    target_local = torch.tensor(target_offset, device=env.device, dtype=torch.float32).expand(count, -1)
    eyes = ee_pos_w + quat_apply(ee_quat_w, eye_local)
    targets = ee_pos_w + quat_apply(ee_quat_w, target_local)
    camera = env.scene[camera_name]
    camera.set_world_poses_from_view(eyes, targets, env_ids=env_ids)
    camera.update(0.0, force_recompute=True)


def randomize_blue_tray_cube_scales(
    env,
    env_ids: torch.Tensor,
    cube_names: tuple[str, ...],
    base_size: float,
    size_range: tuple[float, float],
) -> None:
    """Sample independent XYZ sizes per clone before physics parsing and cache them for layout/metrics."""
    import isaaclab.sim as sim_utils
    from pxr import Gf, Sdf, UsdGeom, Vt

    if env.sim.is_playing():
        raise RuntimeError("Cube scale randomization must run in USD mode before simulation starts")
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, dtype=torch.long)
    else:
        env_ids = env_ids.cpu().to(dtype=torch.long)
    low, high = (float(value) for value in size_range)
    cache = getattr(env, "_blue_tray_cube_sizes", {})
    with Sdf.ChangeBlock():
        for name in cube_names:
            sampled_sizes = torch.empty((len(env_ids), 3), dtype=torch.float32).uniform_(low, high)
            all_sizes = torch.full((env.num_envs, 3), float(base_size), dtype=torch.float32)
            for row, env_id_tensor in enumerate(env_ids):
                env_id = int(env_id_tensor.item())
                prim_path = sim_utils.find_matching_prim_paths(env.scene[name].cfg.prim_path)[env_id]
                prim_spec = Sdf.CreatePrimInLayer(env.sim.stage.GetRootLayer(), prim_path)
                scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
                has_scale = scale_spec is not None
                if not has_scale:
                    scale_spec = Sdf.AttributeSpec(prim_spec, prim_path + ".xformOp:scale", Sdf.ValueTypeNames.Double3)
                scale = sampled_sizes[row] / float(base_size)
                scale_spec.default = Gf.Vec3f(*(float(value) for value in scale.tolist()))
                if not has_scale:
                    order_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
                    if order_spec is None:
                        order_spec = Sdf.AttributeSpec(
                            prim_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
                        )
                    order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])
                all_sizes[env_id] = sampled_sizes[row]
            cache[name] = all_sizes.to(env.device)
    env._blue_tray_cube_sizes = cache


def reset_blue_tray_layout(
    env,
    env_ids: torch.Tensor,
    blue_cube_names: tuple[str, ...],
    red_cube_names: tuple[str, ...],
    cube_sizes: tuple[tuple[float, float, float], ...],
    tray_name: str,
    tray_size: tuple[float, float, float],
    tray_x: float,
    tray_z: float,
    min_spacing: float,
    workspace_x_bounds: tuple[float, float],
    workspace_y_bounds: tuple[float, float],
    workspace_radius_max: float,
    target_workspace_bins: tuple[int, int],
) -> None:
    """Match the recorded training layout distribution exactly."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)

    names = (*blue_cube_names, *red_cube_names)
    size_cache = getattr(env, "_blue_tray_cube_sizes", {})
    poses_by_name = {name: torch.zeros((len(env_ids), 7), device=env.device, dtype=torch.float32) for name in names}
    tray_poses = torch.zeros((len(env_ids), 7), device=env.device, dtype=torch.float32)
    fixed_trays = getattr(env, "_blue_tray_fixed_tray_poses", None)
    if fixed_trays is None:
        fixed_trays = torch.full((env.num_envs, 3), float("nan"), device=env.device, dtype=torch.float32)
    layout_reset_counts = getattr(env, "_blue_tray_layout_reset_counts", None)
    if layout_reset_counts is None:
        layout_reset_counts = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    x_min, x_max = (float(value) for value in workspace_x_bounds)
    y_min, y_max = (float(value) for value in workspace_y_bounds)
    tray_half_x = 0.5 * float(tray_size[0])
    tray_half_y = 0.5 * float(tray_size[1])

    for row, env_id_tensor in enumerate(env_ids):
        env_id = int(env_id_tensor.item())
        if torch.isnan(fixed_trays[env_id, 0]):
            tray_positive = bool(torch.randint(0, 2, (1,), device=env.device).item())
            y_mid = 0.5 * (y_min + y_max)
            tray_y_bounds = (y_mid + min_spacing, y_max) if tray_positive else (y_min, y_mid - min_spacing)
            for _ in range(512):
                tray_y = float(
                    torch.empty(1, device=env.device)
                    .uniform_(tray_y_bounds[0] + tray_half_y, tray_y_bounds[1] - tray_half_y)
                    .item()
                )
                if math.hypot(tray_x, tray_y) <= workspace_radius_max:
                    break
            else:
                raise RuntimeError("Could not sample a reachable fixed tray pose")
            fixed_trays[env_id] = torch.tensor((tray_x, tray_y, tray_z), device=env.device)

        tray_x, tray_y, fixed_tray_z = (float(value) for value in fixed_trays[env_id].tolist())
        tray_poses[row, :3] = torch.tensor((tray_x, tray_y, fixed_tray_z), device=env.device)
        tray_poses[row, 6] = 1.0

        sampled_objects = None
        x_bins, y_bins = (int(value) for value in target_workspace_bins)
        layout_ordinal = int(layout_reset_counts[env_id].item())
        y_stride = max(1, y_bins // max(1, len(blue_cube_names)))
        x_stride = max(1, x_bins // max(1, len(blue_cube_names)))
        x_multiplier = max(1, x_bins - 1)
        while math.gcd(x_multiplier, x_bins) != 1:
            x_multiplier -= 1
        y_multiplier = max(1, y_bins - 1)
        while math.gcd(y_multiplier, y_bins) != 1:
            y_multiplier -= 1
        seed_value = int(torch.initial_seed()) + env_id * 97
        x_offset = seed_value % x_bins
        y_offset = (seed_value // max(1, x_bins)) % y_bins
        blue_indices = {name: index for index, name in enumerate(blue_cube_names)}

        for layout_attempt in range(128):
            # Keep the full training clearance from the tray on every retry.
            # Only inter-object spacing is relaxed for rare dense five-object
            # layouts, matching the generator's ability to retry the layout.
            layout_y_bounds = (y_min, y_max)
            if layout_attempt < 64:
                object_spacing = min_spacing
            elif layout_attempt < 96:
                object_spacing = 0.5 * min_spacing
            else:
                object_spacing = 0.0
            occupied: list[tuple[float, float, float, float]] = [
                (tray_x, tray_y, tray_half_x, tray_half_y)
            ]
            spread_refs: list[tuple[float, float]] = []
            candidate_objects: list[tuple[str, tuple[float, float, float], float, float, float]] = []
            for name, fallback_size in zip(names, cube_sizes, strict=True):
                cached = size_cache.get(name)
                size = fallback_size if cached is None else tuple(float(value) for value in cached[env_id].tolist())
                half = 0.5 * math.hypot(size[0], size[1])

                def overlaps_existing(x: float, y: float) -> bool:
                    return any(
                        abs(x - ox)
                        < half + ohx + (min_spacing if occupied_index == 0 else object_spacing)
                        and abs(y - oy)
                        < half + ohy + (min_spacing if occupied_index == 0 else object_spacing)
                        for occupied_index, (ox, oy, ohx, ohy) in enumerate(occupied)
                    )

                selected_xy: tuple[float, float] | None = None
                if name in blue_indices:
                    blue_index = blue_indices[name]
                    logical_y_start = (
                        layout_ordinal + blue_index * y_stride
                    ) % y_bins
                    logical_x_start = (
                        layout_ordinal // y_bins + blue_index * x_stride
                    ) % x_bins
                    usable_x_min = x_min + half
                    usable_x_max = x_max - half
                    usable_y_min = layout_y_bounds[0] + half
                    usable_y_max = layout_y_bounds[1] - half
                    for y_step in range(y_bins):
                        y_index = (
                            (logical_y_start + y_step) * y_multiplier + y_offset
                        ) % y_bins
                        for x_step in range(x_bins):
                            x_index = (
                                (logical_x_start + x_step) * x_multiplier + x_offset
                            ) % x_bins
                            cell_x_min = usable_x_min + (usable_x_max - usable_x_min) * x_index / x_bins
                            cell_x_max = usable_x_min + (usable_x_max - usable_x_min) * (x_index + 1) / x_bins
                            cell_y_min = usable_y_min + (usable_y_max - usable_y_min) * y_index / y_bins
                            cell_y_max = usable_y_min + (usable_y_max - usable_y_min) * (y_index + 1) / y_bins
                            for _ in range(64):
                                x = float(
                                    torch.empty(1, device=env.device)
                                    .uniform_(cell_x_min, cell_x_max)
                                    .item()
                                )
                                y = float(
                                    torch.empty(1, device=env.device)
                                    .uniform_(cell_y_min, cell_y_max)
                                    .item()
                                )
                                if math.hypot(x, y) > workspace_radius_max or overlaps_existing(x, y):
                                    continue
                                selected_xy = (x, y)
                                break
                            if selected_xy is not None:
                                break
                        if selected_xy is not None:
                            break
                else:
                    candidates: list[tuple[float, float, float]] = []
                    for _ in range(512):
                        x = float(torch.empty(1, device=env.device).uniform_(x_min + half, x_max - half).item())
                        y = float(
                            torch.empty(1, device=env.device)
                            .uniform_(layout_y_bounds[0] + half, layout_y_bounds[1] - half)
                            .item()
                        )
                        if math.hypot(x, y) > workspace_radius_max or overlaps_existing(x, y):
                            continue
                        refs = spread_refs if spread_refs else [(ox, oy) for ox, oy, _, _ in occupied]
                        spread_score = min(math.hypot(x - px, y - py) for px, py in refs)
                        candidates.append((spread_score, x, y))
                    if candidates:
                        candidates.sort(reverse=True)
                        top_count = max(
                            1,
                            min(len(candidates), max(16, math.ceil(len(candidates) * 0.35))),
                        )
                        _, x, y = candidates[
                            int(torch.randint(top_count, (1,), device=env.device).item())
                        ]
                        selected_xy = (x, y)

                if selected_xy is None:
                    break
                x, y = selected_xy
                occupied.append((x, y, half, half))
                spread_refs.append((x, y))
                yaw = float(torch.empty(1, device=env.device).uniform_(-math.pi, math.pi).item())
                candidate_objects.append((name, size, x, y, yaw))
            if len(candidate_objects) == len(names):
                # Fail closed if a future sampler change ever puts an object in
                # or too close to the tray footprint.
                for _name, size, x, y, _yaw in candidate_objects:
                    half = 0.5 * math.hypot(size[0], size[1])
                    if (
                        abs(x - tray_x) < tray_half_x + half + min_spacing
                        and abs(y - tray_y) < tray_half_y + half + min_spacing
                    ):
                        raise RuntimeError(
                            f"Arena sampled object {_name} inside the tray-clear envelope"
                        )
                sampled_objects = candidate_objects
                break
        if sampled_objects is None:
            raise RuntimeError("Could not sample a complete tray-clear reset layout after 128 attempts")
        layout_reset_counts[env_id] += 1

        for name, size, x, y, yaw in sampled_objects:
            poses_by_name[name][row, :3] = torch.tensor((x, y, 0.5 * size[2] + 0.001), device=env.device)
            poses_by_name[name][row, 3:7] = torch.tensor(
                (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)), device=env.device
            )

    env._blue_tray_fixed_tray_poses = fixed_trays
    env._blue_tray_layout_reset_counts = layout_reset_counts
    env_origins = env.scene.env_origins[env_ids]
    for name in names:
        poses_by_name[name][:, :3] += env_origins
        asset = env.scene[name]
        asset.write_root_pose_to_sim(poses_by_name[name], env_ids=env_ids)
        asset.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids)

    tray_poses[:, :3] += env_origins
    tray = env.scene[tray_name]
    tray.write_root_pose_to_sim(tray_poses, env_ids=env_ids)
    tray.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids)

    reset_pose_cache = getattr(env, "_blue_tray_reset_object_poses", {})
    for name in names:
        cached_poses = reset_pose_cache.setdefault(
            name, torch.zeros((env.num_envs, 7), device=env.device, dtype=torch.float32)
        )
        cached_poses[env_ids] = poses_by_name[name]
    env._blue_tray_reset_object_poses = reset_pose_cache


def prepare_blue_tray_policy_episode(env, env_ids: torch.Tensor | None) -> dict:
    """Reproduce the unrecorded generator pre-roll before policy inference starts."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)
    if len(env_ids) != env.num_envs:
        raise RuntimeError("Blue-tray episode preparation requires all vector environments to reset together")

    cfg = env.cfg
    actions = torch.zeros(env.action_space.shape, device=env.device, dtype=torch.float32)
    actions[:, 6] = float(cfg.policy_prepare_gripper_command)
    observation = None

    def step_open(step_count: int) -> None:
        nonlocal observation
        for _ in range(step_count):
            observation, _, terminated, truncated, _ = env.step(actions)
            if terminated.any() or truncated.any():
                raise RuntimeError("Environment terminated during unrecorded policy episode preparation")

    # The generator first lets contacts settle, then restores exact upright object poses.
    step_open(int(cfg.policy_prepare_settle_steps))
    reset_pose_cache = getattr(env, "_blue_tray_reset_object_poses", None)
    if not reset_pose_cache:
        raise RuntimeError("Blue-tray reset poses were not cached before episode preparation")
    zero_velocity = torch.zeros((len(env_ids), 6), device=env.device, dtype=torch.float32)
    for name, cached_poses in reset_pose_cache.items():
        asset = env.scene[name]
        asset.write_root_pose_to_sim(cached_poses[env_ids], env_ids=env_ids)
        asset.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)
    env.sim.forward()
    step_open(int(cfg.policy_prepare_stabilize_steps))

    ee_frame = env.scene["ee_frame"]

    def current_pose() -> tuple[torch.Tensor, torch.Tensor]:
        positions = ee_frame.data.target_pos_w[env_ids, 0] - env.scene.env_origins[env_ids]
        quaternions = ee_frame.data.target_quat_w[env_ids, 0]
        return positions, quaternions

    # Translate only: the validated floor-facing gripper orientation is preserved.
    if cfg.randomize_policy_start_pose:
        targets = torch.empty((len(env_ids), 3), device=env.device, dtype=torch.float32)
        ranges = (cfg.policy_start_ee_x_range, cfg.policy_start_ee_y_range, cfg.policy_start_ee_z_range)
        for row in range(len(env_ids)):
            for _ in range(256):
                candidate = torch.tensor(
                    [float(torch.empty(1, device=env.device).uniform_(*axis_range).item()) for axis_range in ranges],
                    device=env.device,
                    dtype=torch.float32,
                )
                radius = float(torch.linalg.vector_norm(candidate[:2]).item())
                if cfg.policy_start_ee_radius_min <= radius <= cfg.policy_start_ee_radius_max:
                    targets[row] = candidate
                    break
            else:
                raise RuntimeError("Could not sample a reachable randomized policy start EEF target")
    else:
        targets, _ = current_pose()
        targets = targets.clone()

    def tool_down_tilt_deg(quaternions: torch.Tensor) -> torch.Tensor:
        # Preserve the generator historical quaternion convention exactly.
        local_z_world_z = 1.0 - 2.0 * (quaternions[:, 0].square() + quaternions[:, 1].square())
        down_alignment = (-local_z_world_z).clamp(-1.0, 1.0)
        return torch.rad2deg(torch.acos(down_alignment))

    _, initial_quaternions = current_pose()
    if (tool_down_tilt_deg(initial_quaternions) > cfg.policy_start_pose_max_tilt_deg).any():
        raise RuntimeError("Initial tool pose exceeds the generator floor-facing tilt limit")

    reached = torch.zeros(len(env_ids), device=env.device, dtype=torch.bool)
    for _ in range(int(cfg.policy_start_pose_timeout_steps)):
        positions, quaternions = current_pose()
        if (tool_down_tilt_deg(quaternions) > cfg.policy_start_pose_max_tilt_deg).any():
            raise RuntimeError("Tool pose exceeded the generator floor-facing tilt limit during pre-roll")
        delta = targets - positions
        distance = torch.linalg.vector_norm(delta, dim=-1)
        reached = distance <= float(cfg.policy_start_pose_tolerance)
        if reached.all():
            break
        step = delta.clone()
        moving = torch.logical_not(reached)
        scale = (float(cfg.policy_start_pose_step) / distance[moving].clamp_min(1.0e-8)).clamp_max(1.0)
        step[moving] *= scale[:, None]
        step[reached] = 0.0
        move_actions = actions.clone()
        move_actions[env_ids, :3] = step
        observation, _, terminated, truncated, _ = env.step(move_actions)
        if terminated.any() or truncated.any():
            raise RuntimeError("Environment terminated while preparing randomized policy start pose")

    step_open(int(cfg.policy_start_pose_hold_steps))
    positions, final_quaternions = current_pose()
    final_error = torch.linalg.vector_norm(targets - positions, dim=-1)
    final_tilt = tool_down_tilt_deg(final_quaternions)
    if (final_tilt > cfg.policy_start_pose_max_tilt_deg).any():
        raise RuntimeError("Final tool pose exceeds the generator floor-facing tilt limit")
    print(
        "[POLICY-START] "
        f"mode={'randomized' if cfg.randomize_policy_start_pose else 'default'} "
        f"targets={targets.detach().cpu().tolist()} actual={positions.detach().cpu().tolist()} "
        f"error={final_error.detach().cpu().tolist()} tilt={final_tilt.detach().cpu().tolist()}"
    )

    # Preparation is outside the evaluated episode, just like generation before recording.
    env.episode_length_buf[env_ids] = 0
    env.sim.render()
    update_blue_tray_wrist_camera(
        env, env_ids, "wrist_camera", "ee_frame", (0.10, 0.0, -0.08), (0.0, 0.0, 0.12)
    )
    env.scene["external_camera"].update(0.0, force_recompute=True)
    return env.observation_manager.compute()


def reset_blue_tray_room(
    env,
    env_ids: torch.Tensor,
    background_color_jitter: float,
    local_light_count_range: tuple[int, int],
    local_light_intensity_range: tuple[float, float],
    local_light_radius_range: tuple[float, float],
    full_random_background: bool,
) -> None:
    """Create an isolated room per env and randomize it once per episode reset."""
    import isaaclab.sim as sim_utils
    import omni.usd
    from pxr import Gf, UsdGeom, UsdLux

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)

    stage = omni.usd.get_context().get_stage()
    light_count_low, light_count_high = (int(value) for value in local_light_count_range)
    intensity_low, intensity_high = (float(value) for value in local_light_intensity_range)
    radius_low, radius_high = (float(value) for value in local_light_radius_range)

    for env_id_tensor in env_ids:
        env_id = int(env_id_tensor.item())
        if full_random_background:
            background = torch.rand(3, device=env.device)
        else:
            background = torch.tensor((0.10, 0.12, 0.15), device=env.device)
        background_color = tuple(float(value) for value in background.tolist())

        for face_name, face_size, face_position in ROOM_FACES:
            prim_path = f"/World/envs/env_{env_id}/{face_name}"
            prim = stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                face_cfg = sim_utils.CuboidCfg(
                    size=face_size,
                    semantic_tags=[("class", "background")],
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=background_color, roughness=0.9),
                )
                world_position = tuple(float(value) for value in face_position)
                face_cfg.func(prim_path, face_cfg, translation=world_position)

            shader = stage.GetPrimAtPath(f"{prim_path}/geometry/material/Shader")
            if shader and shader.IsValid():
                diffuse = shader.GetAttribute("inputs:diffuseColor")
                if diffuse and diffuse.IsValid():
                    diffuse.Set(Gf.Vec3f(*background_color))

        active_light_count = int(torch.randint(light_count_low, light_count_high + 1, (1,), device=env.device).item())
        for light_id in range(light_count_high):
            light_path = f"/World/envs/env_{env_id}/vector_light_{light_id}"
            light_prim = stage.GetPrimAtPath(light_path)
            enabled = light_id < active_light_count
            if enabled:
                log_intensity = torch.empty(1, device=env.device).uniform_(
                    math.log(intensity_low), math.log(intensity_high)
                )
                intensity = float(torch.exp(log_intensity).item())
                palette_id = int(torch.randint(len(LIGHT_PALETTE), (1,), device=env.device).item())
                light_color_tensor = torch.tensor(LIGHT_PALETTE[palette_id], device=env.device)
                light_color_tensor += torch.empty(3, device=env.device).uniform_(-0.08, 0.08)
                radius = float(torch.empty(1, device=env.device).uniform_(radius_low, radius_high).item())
                light_color = tuple(float(value) for value in light_color_tensor.clamp_(0.0, 1.0).tolist())
                local_position = (
                    float(torch.empty(1, device=env.device).uniform_(0.10, 1.10).item()),
                    float(torch.empty(1, device=env.device).uniform_(-0.90, 0.90).item()),
                    float(torch.empty(1, device=env.device).uniform_(0.80, 1.80).item()),
                )
            else:
                intensity = 0.0
                light_color = (1.0, 1.0, 1.0)
                radius = 0.12
                local_position = (0.5, 0.0, 1.4)
            world_position = local_position

            if not light_prim or not light_prim.IsValid():
                light_cfg = sim_utils.SphereLightCfg(
                    intensity=intensity, color=light_color, radius=radius, normalize=False
                )
                light_cfg.func(light_path, light_cfg, translation=world_position)
                light_prim = stage.GetPrimAtPath(light_path)
            if light_prim and light_prim.IsValid():
                if light_prim.IsInstanceable():
                    light_prim.SetInstanceable(False)
                light = UsdLux.SphereLight(light_prim)
                light.GetIntensityAttr().Set(intensity)
                light.GetColorAttr().Set(Gf.Vec3f(*light_color))
                light.GetRadiusAttr().Set(radius)
                translate_ops = [
                    op
                    for op in UsdGeom.Xformable(light_prim).GetOrderedXformOps()
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
                ]
                if translate_ops:
                    translate_ops[0].Set(Gf.Vec3d(*world_position))


@dataclass
class BlueCubeTrayEnvironmentCfg(ArenaEnvironmentCfg):
    enable_cameras: bool = False
    embodiment: str = "franka_ik"
    num_blue_cubes: int = 3
    num_red_cubes: int = 2
    cube_size: float = 0.05
    # The completed training run recorded 5 cm physical cubes. Its intended
    # per-clone USD scale event remained at unit scale, so evaluation must not
    # introduce unseen 6.5 cm cuboids.
    cube_size_range: list[float] = field(default_factory=lambda: [0.05, 0.05])
    cube_mass_range: list[float] = field(default_factory=lambda: [0.035, 0.075])
    friction_range: list[float] = field(default_factory=lambda: [0.45, 1.10])
    restitution_range: list[float] = field(default_factory=lambda: [0.0, 0.12])
    tray_size: list[float] = field(default_factory=lambda: [0.22, 0.18, 0.025])
    tray_x: float = TRAINING_TRAY_X
    tray_z: float = 0.013
    min_spawn_spacing: float = 0.04
    workspace_x_bounds: list[float] = field(default_factory=lambda: list(WORKSPACE_X_BOUNDS))
    workspace_y_bounds: list[float] = field(default_factory=lambda: list(WORKSPACE_Y_BOUNDS))
    workspace_radius_max: float = WORKSPACE_RADIUS_MAX
    target_workspace_bins: list[int] = field(default_factory=lambda: [4, 6])
    episode_length_s: float = 75.0
    randomize_room_appearance: bool = True
    background_color_jitter: float = 0.06
    local_light_count_range: list[int] = field(default_factory=lambda: [3, 3])
    local_light_intensity_range: list[float] = field(default_factory=lambda: [8000.0, 65000.0])
    local_light_radius_range: list[float] = field(default_factory=lambda: [0.05, 0.20])
    policy_prepare_settle_steps: int = 72
    policy_prepare_stabilize_steps: int = 14
    policy_prepare_gripper_command: float = 1.0
    randomize_policy_start_pose: bool = False
    policy_start_ee_x_range: tuple[float, float] = (0.36, 0.70)
    policy_start_ee_y_range: tuple[float, float] = (-0.34, 0.34)
    policy_start_ee_z_range: tuple[float, float] = (0.25, 0.55)
    policy_start_ee_radius_min: float = 0.40
    policy_start_ee_radius_max: float = 0.72
    policy_start_pose_step: float = 0.03
    policy_start_pose_tolerance: float = 0.012
    policy_start_pose_timeout_steps: int = 480
    policy_start_pose_hold_steps: int = 12
    policy_start_pose_max_tilt_deg: float = 60.0

    def __post_init__(self) -> None:
        assert 1 <= self.num_blue_cubes <= 3
        assert 0 <= self.num_red_cubes <= 3
        assert self.cube_size > 0.0
        assert len(self.cube_size_range) == 2 and 0.0 < self.cube_size_range[0] <= self.cube_size_range[1]
        assert len(self.cube_mass_range) == 2 and 0.0 < self.cube_mass_range[0] <= self.cube_mass_range[1]
        assert len(self.friction_range) == 2 and 0.0 <= self.friction_range[0] <= self.friction_range[1]
        assert len(self.restitution_range) == 2 and 0.0 <= self.restitution_range[0] <= self.restitution_range[1]
        assert len(self.tray_size) == 3 and self.tray_z > 0.0
        assert len(self.workspace_x_bounds) == 2 and self.workspace_x_bounds[0] < self.workspace_x_bounds[1]
        assert len(self.workspace_y_bounds) == 2 and self.workspace_y_bounds[0] < self.workspace_y_bounds[1]
        assert self.workspace_radius_max > 0.0
        assert len(self.target_workspace_bins) == 2 and all(
            int(value) >= 1 for value in self.target_workspace_bins
        )
        assert math.prod(int(value) for value in self.target_workspace_bins) >= self.num_blue_cubes
        assert 0.0 <= self.background_color_jitter <= 1.0
        assert len(self.local_light_count_range) == 2
        assert 1 <= self.local_light_count_range[0] <= self.local_light_count_range[1]
        assert len(self.local_light_intensity_range) == 2
        assert 0.0 < self.local_light_intensity_range[0] <= self.local_light_intensity_range[1]
        assert len(self.local_light_radius_range) == 2
        assert 0.0 < self.local_light_radius_range[0] <= self.local_light_radius_range[1]
        assert self.policy_prepare_settle_steps >= 0 and self.policy_prepare_stabilize_steps >= 0
        assert len(self.policy_start_ee_x_range) == len(self.policy_start_ee_y_range) == 2
        assert len(self.policy_start_ee_z_range) == 2
        assert 0.0 < self.policy_start_ee_radius_min <= self.policy_start_ee_radius_max
        assert self.policy_start_pose_step > 0.0 and self.policy_start_pose_tolerance > 0.0
        assert self.policy_start_pose_timeout_steps > 0 and self.policy_start_pose_hold_steps >= 0
        assert 0.0 < self.policy_start_pose_max_tilt_deg <= 75.0


@register_environment
class BlueCubeTrayEnvironment(ArenaEnvironmentFactory[BlueCubeTrayEnvironmentCfg]):
    """Registered Arena evaluation environment for the generated Franka task."""

    name = "blue_cube_tray"
    _legacy_argparse_cfg_type = BlueCubeTrayEnvironmentCfg

    def build(self, cfg: BlueCubeTrayEnvironmentCfg) -> IsaacLabArenaEnvironment:
        import isaaclab.envs.mdp as mdp_isaac_lab

        # Environment discovery happens before SimulationApp starts. Keep all
        # Isaac/pxr-dependent imports and classes deferred until build time.
        import isaaclab.sim as sim_utils
        from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
        from isaaclab.sensors import CameraCfg
        from isaaclab.utils.configclass import configclass
        from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.blue_cube_tray_task import BlueCubeTrayTask
        from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
        from isaaclab_arena.utils.cameras import ArenaCameraCfg
        from isaaclab_arena.utils.configclass import make_configclass
        from isaaclab_arena.utils.pose import Pose

        class ProceduralCuboid(Object):
            """Arena object backed by one procedural cuboid with an explicit AABB."""

            def __init__(
                self,
                name: str,
                size: tuple[float, float, float],
                color: tuple[float, float, float],
                initial_pose: Pose,
                *,
                kinematic: bool = False,
                mass: float = 0.06,
                visual_only: bool = False,
            ):
                self.size = tuple(float(value) for value in size)
                visual_material = sim_utils.PreviewSurfaceCfg(
                    diffuse_color=color, roughness=0.9 if visual_only else 0.55
                )
                if visual_only:
                    spawner = sim_utils.CuboidCfg(
                        size=self.size,
                        semantic_tags=[("class", "background")],
                        visual_material=visual_material,
                    )
                    object_type = ObjectType.BASE
                else:
                    spawner = sim_utils.CuboidCfg(
                        size=self.size,
                        semantic_tags=[("class", name.rsplit("_", 1)[0])],
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            kinematic_enabled=kinematic,
                            solver_position_iteration_count=16,
                            solver_velocity_iteration_count=1,
                        ),
                        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005),
                        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                        physics_material=sim_utils.RigidBodyMaterialCfg(
                            static_friction=0.70,
                            dynamic_friction=0.58,
                            restitution=0.04,
                        ),
                        visual_material=visual_material,
                    )
                    object_type = ObjectType.RIGID
                super().__init__(
                    name=name,
                    prim_path=f"{{ENV_REGEX_NS}}/{name}",
                    object_type=object_type,
                    spawner_cfg=spawner,
                    initial_pose=initial_pose,
                )

            def get_bounding_box(self) -> AxisAlignedBoundingBox:
                half = tuple(0.5 * value for value in self.size)
                return AxisAlignedBoundingBox(tuple(-value for value in half), half)

            def get_world_bounding_box(self) -> AxisAlignedBoundingBox:
                bbox = self.get_bounding_box()
                pose = self._get_initial_pose_as_pose()
                return bbox if pose is None else bbox.translated(pose.position_xyz)

            def get_corners(self, pos: torch.Tensor) -> torch.Tensor:
                return self.get_bounding_box().get_corners_at(pos)

        class RandomizeCubePhysicsOnce(ManagerTermBase):
            """Sample per-worker cube physics once, matching vector generation."""

            def __init__(self, cfg, env):
                super().__init__(cfg, env)
                params = cfg.params
                asset_cfg = params["asset_cfg"]
                mass_cfg = EventTermCfg(
                    func=mdp_isaac_lab.randomize_rigid_body_mass,
                    mode="reset",
                    params={
                        "asset_cfg": asset_cfg,
                        "mass_distribution_params": params["mass_distribution_params"],
                        "operation": "abs",
                        "distribution": "uniform",
                        "recompute_inertia": True,
                    },
                )
                material_cfg = EventTermCfg(
                    func=mdp_isaac_lab.randomize_rigid_body_material,
                    mode="reset",
                    params={
                        "asset_cfg": asset_cfg,
                        "static_friction_range": params["static_friction_range"],
                        "dynamic_friction_range": params["dynamic_friction_range"],
                        "restitution_range": params["restitution_range"],
                        "num_buckets": 64,
                        "make_consistent": True,
                    },
                )
                self._mass_term = mdp_isaac_lab.randomize_rigid_body_mass(mass_cfg, env)
                self._material_term = mdp_isaac_lab.randomize_rigid_body_material(material_cfg, env)
                self._initialized = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

            def __call__(
                self,
                env,
                env_ids,
                asset_cfg,
                mass_distribution_params,
                static_friction_range,
                dynamic_friction_range,
                restitution_range,
            ):
                if env_ids is None:
                    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
                pending = env_ids[torch.logical_not(self._initialized[env_ids])]
                if len(pending) == 0:
                    return
                self._mass_term(
                    env, pending, asset_cfg, mass_distribution_params, "abs", "uniform", True
                )
                self._material_term(
                    env,
                    pending,
                    static_friction_range,
                    dynamic_friction_range,
                    restitution_range,
                    64,
                    asset_cfg,
                    True,
                )
                self._initialized[pending] = True

        class RandomizeSceneMaterials(ManagerTermBase):
            """Keep room/object colors fixed within an episode and resample them on reset."""

            def __init__(self, cfg, env):
                super().__init__(cfg, env)
                import carb.tokens
                import omni.replicator.core as rep
                from isaacsim.core.experimental.utils.app import enable_extension
                from pxr import Gf, Sdf, UsdShade

                enable_extension("omni.replicator.core")
                self._rep = rep
                env.sim.stage.DefinePrim("/World/Looks", "Scope")
                self._material_batches: dict[int, dict[str, list]] = {}
                self._table_shaders: dict[int, object] = {}
                omni_pbr_mdl = carb.tokens.get_tokens_interface().resolve("${kit}/mdl/core/Base/OmniPBR.mdl")
                blue_names = tuple(cfg.params["blue_names"])
                red_names = tuple(cfg.params["red_names"])
                tray_name = cfg.params["tray_name"]
                table_name = cfg.params["table_name"]

                def make_batch(env_id: int, role: str, prims: list):
                    valid_prims = [prim for prim in prims if prim and prim.IsValid()]
                    for prim in valid_prims:
                        if prim.IsInstanceable():
                            prim.SetInstanceable(False)
                    if not valid_prims:
                        raise RuntimeError(f"No visual prims found for {role} in env {env_id}")
                    return rep.functional.create_batch.material(
                        mdl=omni_pbr_mdl,
                        bind_prims=valid_prims,
                        count=1,
                        name=f"SceneMaterial_{env_id}_{role}",
                        parent="/World/Looks",
                        project_uvw=True,
                    )

                for env_id in range(env.num_envs):
                    root = f"/World/envs/env_{env_id}"
                    groups: dict[str, list] = {}
                    groups["room"] = [
                        make_batch(
                            env_id,
                            f"room_{face_name}",
                            [env.sim.stage.GetPrimAtPath(f"{root}/{face_name}/geometry")],
                        )
                        for face_name, _, _ in ROOM_FACES
                    ]
                    for role, names in (("blue", blue_names), ("red", red_names)):
                        groups[role] = [
                            make_batch(
                                env_id,
                                f"{role}_{index}",
                                [env.sim.stage.GetPrimAtPath(f"{root}/{name}/geometry")],
                            )
                            for index, name in enumerate(names)
                        ]
                    groups["tray"] = [
                        make_batch(env_id, "tray", [env.sim.stage.GetPrimAtPath(f"{root}/{tray_name}/geometry")])
                    ]
                    table_root = env.sim.stage.GetPrimAtPath(f"{root}/{table_name}")
                    material = UsdShade.Material.Define(env.sim.stage, f"/World/Looks/TableMaterial_{env_id}")
                    shader = UsdShade.Shader.Define(env.sim.stage, f"/World/Looks/TableMaterial_{env_id}/Shader")
                    shader.CreateIdAttr("UsdPreviewSurface")
                    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*TABLE_PALETTE[0]))
                    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
                    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                    UsdShade.MaterialBindingAPI.Apply(table_root).Bind(
                        material, UsdShade.Tokens.strongerThanDescendants
                    )
                    self._table_shaders[env_id] = shader
                    self._material_batches[env_id] = groups

            def __call__(
                self,
                env,
                env_ids,
                blue_names,
                red_names,
                tray_name,
                table_name,
                background_colors,
                background_jitter: float,
                full_random_background: bool,
                blue_color,
                red_color,
                cube_jitter: float,
                tray_colors,
                table_colors,
                surface_jitter: float,
            ):
                import numpy as np

                from pxr import Gf

                if env_ids is None:
                    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

                def sampled_values(palette, jitter: float):
                    index = int(torch.randint(len(palette), (1,), device=env.device).item())
                    color = torch.tensor(palette[index], device=env.device, dtype=torch.float32)
                    color += torch.empty(3, device=env.device).uniform_(-jitter, jitter)
                    return np.asarray([color.clamp_(0.0, 1.0).cpu().tolist()], dtype=np.float32)

                for env_id_tensor in env_ids:
                    env_id = int(env_id_tensor.item())
                    room_values = (
                        np.asarray([torch.rand(3, device=env.device).cpu().tolist()], dtype=np.float32)
                        if full_random_background
                        else sampled_values(background_colors, background_jitter)
                    )
                    role_values = {
                        "room": room_values,
                        "blue": sampled_values((blue_color,), cube_jitter),
                        "red": sampled_values((red_color,), cube_jitter),
                        "tray": sampled_values(tray_colors, surface_jitter),
                        "table": sampled_values(table_colors, surface_jitter),
                    }
                    for role, batches in self._material_batches[env_id].items():
                        for material_batch in batches:
                            self._rep.functional.modify.attribute(
                                material_batch, "diffuse_color_constant", role_values[role]
                            )

                    table_rgb = role_values["table"][0].tolist()
                    self._table_shaders[env_id].GetInput("diffuseColor").Set(
                        Gf.Vec3f(*(float(value) for value in table_rgb))
                    )

        @configclass
        class BlueTrayFrankaCameraCfg(ArenaCameraCfg):
            """The two RGB views used to train the Franka GR00T checkpoint."""

            external_camera: CameraCfg = CameraCfg(
                prim_path="{ENV_REGEX_NS}/external_camera",
                update_period=1.0 / CAMERA_FPS,
                height=CAMERA_HEIGHT,
                width=CAMERA_WIDTH,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=EXTERNAL_CAMERA_FOCAL_LENGTH,
                    focus_distance=400.0,
                    horizontal_aperture=CAMERA_HORIZONTAL_APERTURE,
                    clipping_range=(0.05, 20.0),
                ),
                # eye=(1.3,-1.3,1.0), target=(0.35,0.0,0.25), OpenGL frame.
                offset=CameraCfg.OffsetCfg(
                    pos=(1.3, -1.3, 1.0),
                    rot=(0.5109, 0.1668, 0.2617, 0.8016),
                    convention="opengl",
                ),
            )
            wrist_camera: CameraCfg = CameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_camera",
                update_period=1.0 / CAMERA_FPS,
                height=CAMERA_HEIGHT,
                width=CAMERA_WIDTH,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=WRIST_CAMERA_FOCAL_LENGTH,
                    focus_distance=0.30,
                    horizontal_aperture=CAMERA_HORIZONTAL_APERTURE,
                    clipping_range=(0.02, 5.0),
                ),
                # Source-equivalent mount including the synthetic EE frame z-offset (0.1034 m).
                offset=CameraCfg.OffsetCfg(
                    pos=(0.10, 0.0, 0.0234),
                    rot=(0.6882, 0.6882, 0.1625, 0.1625),
                    convention="opengl",
                ),
            )

        table = self.asset_registry.get_asset_by_name("table")()
        table.usd_path = table.usd_path.replace("/Assets/Isaac/6.0/", "/Assets/Isaac/5.1/")
        table.object_cfg.spawn.usd_path = table.usd_path
        table.set_initial_pose(Pose(position_xyz=(0.5, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.7071068, 0.7071068)))
        dome_light = self.asset_registry.get_asset_by_name("light")()
        dome_light.spawner_cfg.visible_in_primary_ray = False
        dome_light.set_color((0.86, 0.86, 0.86))
        dome_light.set_intensity(900.0)

        size = (cfg.cube_size, cfg.cube_size, cfg.cube_size)
        blue_names = tuple(f"blue_cube_{index}" for index in range(cfg.num_blue_cubes))
        red_names = tuple(f"red_cube_{index}" for index in range(cfg.num_red_cubes))
        all_names = (*blue_names, *red_names)
        cube_sizes = tuple(size for _ in all_names)

        cubes = []
        for index, name in enumerate(all_names):
            color = BLUE if name.startswith("blue") else RED
            cube = ProceduralCuboid(
                name=name,
                size=size,
                color=color,
                initial_pose=Pose(position_xyz=(0.36 + 0.07 * index, -0.20, 0.5 * cfg.cube_size + 0.001)),
            )
            cube.disable_reset_pose()
            cubes.append(cube)

        tray = ProceduralCuboid(
            name="green_tray",
            size=cfg.tray_size,
            color=GREEN,
            initial_pose=Pose(position_xyz=(0.52, 0.18, 0.5 * cfg.tray_size[2])),
            kinematic=True,
            mass=1.0,
        )
        tray.disable_reset_pose()

        room_faces = [
            ProceduralCuboid(
                name=face_name,
                size=face_size,
                color=(0.10, 0.12, 0.15),
                initial_pose=Pose(position_xyz=face_position),
                visual_only=True,
            )
            for face_name, face_size, face_position in ROOM_FACES
        ]
        local_lights = [
            Object(
                name=f"vector_light_{light_id}",
                prim_path=f"{{ENV_REGEX_NS}}/vector_light_{light_id}",
                object_type=ObjectType.BASE,
                spawner_cfg=sim_utils.SphereLightCfg(
                    intensity=45000.0,
                    color=(1.0, 0.96, 0.90),
                    radius=0.12,
                    normalize=False,
                ),
                initial_pose=Pose(position_xyz=(0.5, -0.3, 1.4)),
            )
            for light_id in range(max(cfg.local_light_count_range))
        ]

        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(enable_cameras=cfg.enable_cameras)
        # Match the synthetic generator exactly: stock Isaac Lab Franka asset,
        # fixed tabletop joint state, and no reset-time joint noise.
        robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Training episodes explicitly used the Isaac 5.1 asset tree.
        robot_cfg.spawn.usd_path = robot_cfg.spawn.usd_path.replace("/Assets/Isaac/6.0/", "/Assets/Isaac/5.1/")
        robot_cfg.init_state.joint_pos = TABLETOP_FRANKA_JOINT_POS.copy()
        embodiment.scene_config.robot = robot_cfg
        # Disable Arena's index-ordered pose override; the named joint map above
        # matches the synthetic generator and is robust to USD joint ordering.
        embodiment.event_config.init_franka_arm_pose = None
        embodiment.event_config.randomize_franka_joint_state.params["std"] = 0.0
        embodiment.camera_config = BlueTrayFrankaCameraCfg()

        scene = Scene(
            assets=[
                table,
                dome_light,
                tray,
                *cubes,
                *room_faces,
                *local_lights,
            ]
        )
        event_fields = [
            (
                "randomize_blue_tray_cube_scales",
                EventTermCfg,
                EventTermCfg(
                    func=randomize_blue_tray_cube_scales,
                    mode="usd",
                    params={
                        "cube_names": all_names,
                        "base_size": cfg.cube_size,
                        "size_range": tuple(cfg.cube_size_range),
                    },
                ),
            ),
            (
                "update_blue_tray_wrist_camera",
                EventTermCfg,
                EventTermCfg(
                    func=update_blue_tray_wrist_camera,
                    mode="interval",
                    interval_range_s=(1.0 / CAMERA_FPS, 1.0 / CAMERA_FPS),
                    is_global_time=True,
                    params={
                        "camera_name": "wrist_camera",
                        "ee_frame_name": "ee_frame",
                        "eye_offset": (0.10, 0.0, -0.08),
                        "target_offset": (0.0, 0.0, 0.12),
                    },
                ),
            ),
            (
                "reset_blue_tray_layout",
                EventTermCfg,
                EventTermCfg(
                    func=reset_blue_tray_layout,
                    mode="reset",
                    params={
                        "blue_cube_names": blue_names,
                        "red_cube_names": red_names,
                        "cube_sizes": cube_sizes,
                        "tray_name": tray.name,
                        "tray_size": cfg.tray_size,
                        "tray_x": cfg.tray_x,
                        "tray_z": cfg.tray_z,
                        "min_spacing": cfg.min_spawn_spacing,
                        "workspace_x_bounds": tuple(cfg.workspace_x_bounds),
                        "workspace_y_bounds": tuple(cfg.workspace_y_bounds),
                        "workspace_radius_max": cfg.workspace_radius_max,
                        "target_workspace_bins": tuple(cfg.target_workspace_bins),
                    },
                ),
            ),
            (
                "reset_blue_tray_room",
                EventTermCfg,
                EventTermCfg(
                    func=reset_blue_tray_room,
                    mode="reset",
                    params={
                        "background_color_jitter": (
                            cfg.background_color_jitter if cfg.randomize_room_appearance else 0.0
                        ),
                        "local_light_count_range": (
                            tuple(cfg.local_light_count_range) if cfg.randomize_room_appearance else (3, 3)
                        ),
                        "local_light_intensity_range": (
                            tuple(cfg.local_light_intensity_range)
                            if cfg.randomize_room_appearance
                            else (24000.0, 24000.0)
                        ),
                        "local_light_radius_range": (
                            tuple(cfg.local_light_radius_range) if cfg.randomize_room_appearance else (0.12, 0.12)
                        ),
                        "full_random_background": cfg.randomize_room_appearance,
                    },
                ),
            ),
            (
                "randomize_blue_tray_scene_materials",
                EventTermCfg,
                EventTermCfg(
                    func=RandomizeSceneMaterials,
                    mode="reset",
                    params={
                        "blue_names": blue_names,
                        "red_names": red_names,
                        "tray_name": tray.name,
                        "table_name": table.name,
                        "background_colors": (
                            BACKGROUND_PALETTE if cfg.randomize_room_appearance else ((0.10, 0.12, 0.15),)
                        ),
                        "background_jitter": cfg.background_color_jitter if cfg.randomize_room_appearance else 0.0,
                        "full_random_background": cfg.randomize_room_appearance,
                        "blue_color": (0.03, 0.16, 0.95),
                        "red_color": (0.95, 0.04, 0.03),
                        "cube_jitter": 0.04 if cfg.randomize_room_appearance else 0.0,
                        "tray_colors": TRAY_PALETTE if cfg.randomize_room_appearance else (TRAY_PALETTE[0],),
                        "table_colors": TABLE_PALETTE if cfg.randomize_room_appearance else (TABLE_PALETTE[0],),
                        "surface_jitter": 0.05 if cfg.randomize_room_appearance else 0.0,
                    },
                ),
            ),
        ]
        for name in all_names:
            event_fields.append(
                (
                    f"randomize_{name}_physics_once",
                    EventTermCfg,
                    EventTermCfg(
                        func=RandomizeCubePhysicsOnce,
                        mode="reset",
                        params={
                            "asset_cfg": SceneEntityCfg(name),
                            "mass_distribution_params": tuple(cfg.cube_mass_range),
                            "static_friction_range": tuple(cfg.friction_range),
                            "dynamic_friction_range": tuple(cfg.friction_range),
                            "restitution_range": tuple(cfg.restitution_range),
                        },
                    ),
                )
            )
        if not cfg.enable_cameras:
            event_fields = [field for field in event_fields if field[0] != "update_blue_tray_wrist_camera"]
        events_cfg_type = make_configclass("BlueTrayEventsCfg", event_fields)
        scene.events_cfg = events_cfg_type()

        task = BlueCubeTrayTask(
            blue_cube_names=blue_names,
            red_cube_names=red_names,
            cube_sizes=cube_sizes,
            tray_name=tray.name,
            tray_size=cfg.tray_size,
            episode_length_s=cfg.episode_length_s,
        )

        def configure_timing(env_cfg):
            env_cfg.scene.replicate_physics = False
            env_cfg.sim.dt = 1.0 / 120.0
            env_cfg.decimation = 2
            env_cfg.sim.render_interval = 8
            # Match the generator before recording; task actions still come only from GR00T.
            env_cfg.num_rerenders_on_reset = 1
            env_cfg.policy_episode_prepare_callback = prepare_blue_tray_policy_episode
            env_cfg.policy_prepare_settle_steps = cfg.policy_prepare_settle_steps
            env_cfg.policy_prepare_stabilize_steps = cfg.policy_prepare_stabilize_steps
            env_cfg.policy_prepare_gripper_command = cfg.policy_prepare_gripper_command
            env_cfg.randomize_policy_start_pose = cfg.randomize_policy_start_pose
            env_cfg.policy_start_ee_x_range = tuple(cfg.policy_start_ee_x_range)
            env_cfg.policy_start_ee_y_range = tuple(cfg.policy_start_ee_y_range)
            env_cfg.policy_start_ee_z_range = tuple(cfg.policy_start_ee_z_range)
            env_cfg.policy_start_ee_radius_min = cfg.policy_start_ee_radius_min
            env_cfg.policy_start_ee_radius_max = cfg.policy_start_ee_radius_max
            env_cfg.policy_start_pose_step = cfg.policy_start_pose_step
            env_cfg.policy_start_pose_tolerance = cfg.policy_start_pose_tolerance
            env_cfg.policy_start_pose_timeout_steps = cfg.policy_start_pose_timeout_steps
            env_cfg.policy_start_pose_hold_steps = cfg.policy_start_pose_hold_steps
            env_cfg.policy_start_pose_max_tilt_deg = cfg.policy_start_pose_max_tilt_deg
            return env_cfg

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            env_cfg_callback=configure_timing,
        )
