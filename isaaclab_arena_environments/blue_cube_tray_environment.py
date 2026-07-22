# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Arena environment matching the Franka blue-cuboid-to-green-tray dataset."""

from __future__ import annotations

import math
import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.cameras import ArenaCameraCfg
from isaaclab_arena.utils.configclass import make_configclass
from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


BLUE = (0.0426, 0.1420, 0.9345)
RED = (0.9899, 0.0162, 0.0290)
GREEN = (0.1244, 0.5804, 0.1832)
TABLETOP_FRANKA_POSE = [0.50, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741, 0.040, 0.040]


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
    ):
        self.size = tuple(float(value) for value in size)
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
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.55),
        )
        super().__init__(
            name=name,
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            object_type=ObjectType.RIGID,
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


@configclass
class BlueTrayFrankaCameraCfg(ArenaCameraCfg):
    """The two RGB views used to train the Franka GR00T checkpoint."""

    external_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/external_camera",
        update_period=1.0 / 15.0,
        height=256,
        width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 20.0),
        ),
        # eye=(1.3,-1.3,1.0), target=(0.35,0.0,0.25), OpenGL camera frame.
        offset=CameraCfg.OffsetCfg(
            pos=(1.3, -1.3, 1.0),
            rot=(0.5109, 0.1668, 0.2617, 0.8016),
            convention="opengl",
        ),
    )
    wrist_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_camera",
        update_period=1.0 / 15.0,
        height=256,
        width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=0.30,
            horizontal_aperture=20.955,
            clipping_range=(0.02, 5.0),
        ),
        # Equivalent fixed mount for source eye=(.10,0,-.08), target=(0,0,.12).
        offset=CameraCfg.OffsetCfg(
            pos=(0.10, 0.0, -0.08),
            rot=(0.6882, 0.6882, 0.1625, 0.1625),
            convention="opengl",
        ),
    )


def reset_blue_tray_layout(
    env,
    env_ids: torch.Tensor,
    blue_cube_names: tuple[str, ...],
    red_cube_names: tuple[str, ...],
    cube_sizes: tuple[tuple[float, float, float], ...],
    tray_name: str,
    tray_size: tuple[float, float, float],
    min_spacing: float,
) -> None:
    """Sample a non-overlapping robot-front layout independently for each reset env."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)

    names = (*blue_cube_names, *red_cube_names)
    poses_by_name = {
        name: torch.zeros((len(env_ids), 7), device=env.device, dtype=torch.float32) for name in names
    }
    tray_poses = torch.zeros((len(env_ids), 7), device=env.device, dtype=torch.float32)

    for row in range(len(env_ids)):
        tray_x = float(torch.empty(1, device=env.device).uniform_(0.47, 0.57).item())
        tray_y = float(torch.empty(1, device=env.device).uniform_(0.14, 0.22).item())
        tray_poses[row, :3] = torch.tensor((tray_x, tray_y, 0.5 * tray_size[2]), device=env.device)
        tray_poses[row, 6] = 1.0

        accepted: list[tuple[float, float, float]] = []
        for name, size in zip(names, cube_sizes, strict=True):
            radius = 0.5 * math.hypot(size[0], size[1])
            for _ in range(256):
                x = float(torch.empty(1, device=env.device).uniform_(0.34, 0.68).item())
                y = float(torch.empty(1, device=env.device).uniform_(-0.30, 0.04).item())
                if all(math.hypot(x - px, y - py) >= radius + pr + min_spacing for px, py, pr in accepted):
                    accepted.append((x, y, radius))
                    break
            else:
                raise RuntimeError(f"Could not sample a non-overlapping reset pose for {name}")

            yaw = float(torch.empty(1, device=env.device).uniform_(-math.pi, math.pi).item())
            poses_by_name[name][row, :3] = torch.tensor((x, y, 0.5 * size[2] + 0.001), device=env.device)
            poses_by_name[name][row, 2:7] = torch.tensor(
                (0.5 * size[2] + 0.001, 0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)),
                device=env.device,
            )

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


@dataclass
class BlueCubeTrayEnvironmentCfg(ArenaEnvironmentCfg):
    enable_cameras: bool = False
    embodiment: str = "franka_ik"
    num_blue_cubes: int = 3
    num_red_cubes: int = 2
    cube_size: float = 0.05
    tray_size: tuple[float, float, float] = (0.22, 0.18, 0.025)
    min_spawn_spacing: float = 0.025
    episode_length_s: float = 75.0

    def __post_init__(self) -> None:
        assert 1 <= self.num_blue_cubes <= 3
        assert 0 <= self.num_red_cubes <= 3
        assert self.cube_size > 0.0


@register_environment
class BlueCubeTrayEnvironment(ArenaEnvironmentFactory[BlueCubeTrayEnvironmentCfg]):
    """Registered Arena evaluation environment for the generated Franka task."""

    name = "blue_cube_tray"
    _legacy_argparse_cfg_type = BlueCubeTrayEnvironmentCfg

    def build(self, cfg: BlueCubeTrayEnvironmentCfg) -> IsaacLabArenaEnvironment:
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.blue_cube_tray_task import BlueCubeTrayTask

        table = self.asset_registry.get_asset_by_name("table")()
        table.set_initial_pose(Pose(position_xyz=(0.5, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
        ground = self.asset_registry.get_asset_by_name("ground_plane")()
        ground.set_initial_pose(Pose(position_xyz=(0.0, 0.0, -1.05)))
        dome_light = self.asset_registry.get_asset_by_name("light")()
        dome_light.set_intensity(900.0)
        directional_light = self.asset_registry.get_asset_by_name("directional_light")()

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

        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(enable_cameras=cfg.enable_cameras)
        embodiment.set_initial_joint_pose(TABLETOP_FRANKA_POSE)
        embodiment.camera_config = BlueTrayFrankaCameraCfg()

        scene = Scene(assets=[table, ground, dome_light, directional_light, tray, *cubes])
        events_cfg_type = make_configclass(
            "BlueTrayEventsCfg",
            [
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
                            "min_spacing": cfg.min_spawn_spacing,
                        },
                    ),
                )
            ],
        )
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
            env_cfg.sim.dt = 1.0 / 120.0
            env_cfg.decimation = 2
            env_cfg.sim.render_interval = 8
            return env_cfg

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            env_cfg_callback=configure_timing,
        )
