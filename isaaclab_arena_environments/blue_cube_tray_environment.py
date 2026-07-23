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
    min_spacing: float,
) -> None:
    """Match the generator workspace, reach limit, collision margins, and spread sampling."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)

    names = (*blue_cube_names, *red_cube_names)
    size_cache = getattr(env, "_blue_tray_cube_sizes", {})
    poses_by_name = {name: torch.zeros((len(env_ids), 7), device=env.device, dtype=torch.float32) for name in names}
    tray_poses = torch.zeros((len(env_ids), 7), device=env.device, dtype=torch.float32)

    for row, env_id_tensor in enumerate(env_ids):
        env_id = int(env_id_tensor.item())
        # Generator bounds: x=(0.40, 0.62), y=(0.04, 0.30), minus tray half extents.
        tray_x = float(
            torch.empty(1, device=env.device).uniform_(0.40 + 0.5 * tray_size[0], 0.62 - 0.5 * tray_size[0]).item()
        )
        tray_y = float(
            torch.empty(1, device=env.device).uniform_(0.04 + 0.5 * tray_size[1], 0.30 - 0.5 * tray_size[1]).item()
        )
        tray_poses[row, :3] = torch.tensor((tray_x, tray_y, 0.5 * tray_size[2]), device=env.device)
        tray_poses[row, 6] = 1.0

        occupied: list[tuple[float, float, float, float]] = [(tray_x, tray_y, 0.5 * tray_size[0], 0.5 * tray_size[1])]
        spread_refs: list[tuple[float, float]] = []
        for name, fallback_size in zip(names, cube_sizes, strict=True):
            cached = size_cache.get(name)
            if cached is None:
                size = fallback_size
            else:
                size = tuple(float(value) for value in cached[env_id].tolist())
            half = 0.5 * max(size[0], size[1])
            candidates: list[tuple[float, float, float]] = []
            for _ in range(512):
                x = float(torch.empty(1, device=env.device).uniform_(0.33 + half, 0.62 - half).item())
                y = float(torch.empty(1, device=env.device).uniform_(-0.30 + half, 0.30 - half).item())
                if math.hypot(x, y) > 0.66:
                    continue
                if any(
                    abs(x - ox) < half + ohx + min_spacing and abs(y - oy) < half + ohy + min_spacing
                    for ox, oy, ohx, ohy in occupied
                ):
                    continue
                refs = spread_refs if spread_refs else [(ox, oy) for ox, oy, _, _ in occupied]
                spread_score = min(math.hypot(x - px, y - py) for px, py in refs)
                candidates.append((spread_score, x, y))
            if not candidates:
                raise RuntimeError(f"Could not sample a spread reset pose for {name}")
            candidates.sort(reverse=True)
            top_count = max(1, min(len(candidates), max(8, len(candidates) // 10)))
            choice = int(torch.randint(top_count, (1,), device=env.device).item())
            _, x, y = candidates[choice]
            occupied.append((x, y, half, half))
            spread_refs.append((x, y))

            yaw = float(torch.empty(1, device=env.device).uniform_(-math.pi, math.pi).item())
            poses_by_name[name][row, :3] = torch.tensor((x, y, 0.5 * size[2] + 0.001), device=env.device)
            poses_by_name[name][row, 3:7] = torch.tensor(
                (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)), device=env.device
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


def reset_blue_tray_room(
    env,
    env_ids: torch.Tensor,
    background_color_jitter: float,
    local_light_count_range: tuple[int, int],
    local_light_intensity_range: tuple[float, float],
    local_light_radius_range: tuple[float, float],
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
        palette_index = int(torch.randint(len(BACKGROUND_PALETTE), (1,), device=env.device).item())
        background = torch.tensor(BACKGROUND_PALETTE[palette_index], device=env.device)
        background += torch.empty(3, device=env.device).uniform_(-background_color_jitter, background_color_jitter)
        background_color = tuple(float(value) for value in background.clamp_(0.0, 1.0).tolist())

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
    cube_size_range: list[float] = field(default_factory=lambda: [0.05, 0.065])
    cube_mass_range: list[float] = field(default_factory=lambda: [0.035, 0.075])
    friction_range: list[float] = field(default_factory=lambda: [0.45, 1.10])
    restitution_range: list[float] = field(default_factory=lambda: [0.0, 0.12])
    tray_size: list[float] = field(default_factory=lambda: [0.22, 0.18, 0.025])
    min_spawn_spacing: float = 0.04
    episode_length_s: float = 75.0
    randomize_room_appearance: bool = True
    background_color_jitter: float = 0.06
    local_light_count_range: list[int] = field(default_factory=lambda: [3, 3])
    local_light_intensity_range: list[float] = field(default_factory=lambda: [8000.0, 65000.0])
    local_light_radius_range: list[float] = field(default_factory=lambda: [0.05, 0.20])

    def __post_init__(self) -> None:
        assert 1 <= self.num_blue_cubes <= 3
        assert 0 <= self.num_red_cubes <= 3
        assert self.cube_size > 0.0
        assert len(self.cube_size_range) == 2 and 0.0 < self.cube_size_range[0] <= self.cube_size_range[1]
        assert len(self.cube_mass_range) == 2 and 0.0 < self.cube_mass_range[0] <= self.cube_mass_range[1]
        assert len(self.friction_range) == 2 and 0.0 <= self.friction_range[0] <= self.friction_range[1]
        assert len(self.restitution_range) == 2 and 0.0 <= self.restitution_range[0] <= self.restitution_range[1]
        assert len(self.tray_size) == 3
        assert 0.0 <= self.background_color_jitter <= 1.0
        assert len(self.local_light_count_range) == 2
        assert 1 <= self.local_light_count_range[0] <= self.local_light_count_range[1]
        assert len(self.local_light_intensity_range) == 2
        assert 0.0 < self.local_light_intensity_range[0] <= self.local_light_intensity_range[1]
        assert len(self.local_light_radius_range) == 2
        assert 0.0 < self.local_light_radius_range[0] <= self.local_light_radius_range[1]


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
                    role_values = {
                        "room": sampled_values(background_colors, background_jitter),
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
        ground = self.asset_registry.get_asset_by_name("ground_plane")()
        ground.set_initial_pose(Pose(position_xyz=(0.0, 0.0, -1.05)))
        dome_light = self.asset_registry.get_asset_by_name("light")()
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
                ground,
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
                        "min_spacing": cfg.min_spawn_spacing,
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
            event_fields.extend([
                (
                    f"randomize_{name}_mass",
                    EventTermCfg,
                    EventTermCfg(
                        func=mdp_isaac_lab.randomize_rigid_body_mass,
                        mode="reset",
                        params={
                            "asset_cfg": SceneEntityCfg(name),
                            "mass_distribution_params": tuple(cfg.cube_mass_range),
                            "operation": "abs",
                            "distribution": "uniform",
                            "recompute_inertia": True,
                        },
                    ),
                ),
                (
                    f"randomize_{name}_material",
                    EventTermCfg,
                    EventTermCfg(
                        func=mdp_isaac_lab.randomize_rigid_body_material,
                        mode="reset",
                        params={
                            "asset_cfg": SceneEntityCfg(name),
                            "static_friction_range": tuple(cfg.friction_range),
                            "dynamic_friction_range": tuple(cfg.friction_range),
                            "restitution_range": tuple(cfg.restitution_range),
                            "num_buckets": 64,
                            "make_consistent": True,
                        },
                    ),
                ),
            ])
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
            # Flush Fabric/RTX sensor transforms after reset before policy observation.
            env_cfg.num_rerenders_on_reset = 1
            env_cfg.policy_warmup_steps = 4
            env_cfg.policy_warmup_action = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
            return env_cfg

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            env_cfg_callback=configure_timing,
        )
