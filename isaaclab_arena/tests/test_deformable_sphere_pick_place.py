# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the deformable pick-and-place environment (config + Newton end-to-end smoke)."""

import types

import pytest

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True


def test_deformable_assets_registered() -> None:
    from isaaclab_arena.assets.registries import AssetRegistry

    reg = AssetRegistry()
    assert reg.is_registered("procedural_deformable_sphere")
    assert reg.is_registered("procedural_deformable_cube")


def test_deformable_sphere_cfg_type() -> None:
    """The sphere's object cfg is a per-backend PresetCfg that resolves to a DeformableObjectCfg."""
    from isaaclab.assets import DeformableObjectCfg
    from isaaclab_tasks.utils import PresetCfg
    from isaaclab_tasks.utils.hydra import resolve_presets

    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.utils.pose import Pose

    sphere = AssetRegistry().get_asset_by_name("procedural_deformable_sphere")()
    sphere.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1)))

    assert sphere.object_type == ObjectType.DEFORMABLE
    assert isinstance(sphere.object_cfg, PresetCfg)

    physx_cfg = resolve_presets(sphere.object_cfg, selected=("physx",))
    newton_cfg = resolve_presets(sphere.object_cfg, selected=("newton_mjwarp_vbd",))
    assert isinstance(physx_cfg, DeformableObjectCfg)
    assert isinstance(newton_cfg, DeformableObjectCfg)
    # Initial pose is stamped on every backend variant.
    assert physx_cfg.init_state.pos == (0.4, 0.0, 0.1)
    assert newton_cfg.init_state.pos == (0.4, 0.0, 0.1)
    # A nodal reset event is generated for the deformable.
    assert sphere.get_event_cfg()[1] is not None


def test_deformable_spawn_uses_pretet_usd() -> None:
    """Both backends spawn from the committed pre-tetrahedralized TetMesh USD (no runtime pytetwild)."""
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
    from isaaclab_tasks.utils.hydra import resolve_presets

    from isaaclab_arena.assets.registries import AssetRegistry

    for asset_name, tet_file in (
        ("procedural_deformable_sphere", "procedural_deformable_sphere_tet.usda"),
        ("procedural_deformable_cube", "procedural_deformable_cube_tet.usda"),
    ):
        asset = AssetRegistry().get_asset_by_name(asset_name)()
        for backend in ("physx", "newton_mjwarp_vbd"):
            cfg = resolve_presets(asset.object_cfg, selected=(backend,))
            assert isinstance(cfg.spawn, UsdFileCfg), f"{asset_name}/{backend} is not a UsdFileCfg"
            assert cfg.spawn.usd_path.endswith(tet_file), f"{asset_name}/{backend} not pointing at {tet_file}"


def test_deformable_pick_and_place_task_cfg() -> None:
    """The task builds its termination cfg from the shipped pick/destination assets.

    The background is stubbed to its ``object_min_z`` (the only field the task reads): constructing a
    real ``Background`` eagerly opens its remote USD, which requires a running SimulationApp. The real
    ``maple_table_robolab`` background is exercised end-to-end by ``test_deformable_sphere_droid_newton_smoke``.
    """
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.metrics.success_rate import SuccessRateMetric
    from isaaclab_arena.tasks.deformable_pick_and_place_task import (
        DeformablePickAndPlaceTask,
        DeformablePickAndPlaceTerminationsCfg,
    )

    reg = AssetRegistry()
    sphere = reg.get_asset_by_name("procedural_deformable_sphere")()
    bowl = reg.get_asset_by_name("bowl_ycb_robolab")()
    background = types.SimpleNamespace(name="maple_table_robolab", object_min_z=-0.2)

    task = DeformablePickAndPlaceTask(
        pick_up_object=sphere,
        destination_location=bowl,
        background_scene=background,
    )

    assert task.get_scene_cfg() is None
    assert task.get_events_cfg() is None
    assert isinstance(task.get_termination_cfg(), DeformablePickAndPlaceTerminationsCfg)
    assert isinstance(task.get_metrics()[0], SuccessRateMetric)


def test_deformable_environment_in_cli_registry() -> None:
    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena_environments.cli import ensure_environments_registered

    ensure_environments_registered()
    env_registry = EnvironmentRegistry()
    assert env_registry.is_registered("deformable_sphere_pick_place")
    assert env_registry.get_component_by_name("deformable_sphere_pick_place").name == "deformable_sphere_pick_place"


def test_deformable_physics_backend_selection() -> None:
    """ArenaEnvBuilder rejects the rigid ``newton`` preset and forces replicate off for PhysX deformables."""
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    sphere = AssetRegistry().get_asset_by_name("procedural_deformable_sphere")()
    builder = object.__new__(ArenaEnvBuilder)
    builder.arena_env = types.SimpleNamespace(scene=types.SimpleNamespace(assets={"object": sphere}))

    assert builder._scene_has_deformable_objects() is True

    # Rigid-body Newton MJWarp preset cannot simulate deformables -> explicit error.
    with pytest.raises(NotImplementedError, match="newton_mjwarp_vbd"):
        builder._configure_physics_for_scene(types.SimpleNamespace(), "newton")

    # No preset -> PhysX fallback; PhysX cannot replicate deformables.
    env_cfg = types.SimpleNamespace(scene=types.SimpleNamespace(replicate_physics=True))
    builder._configure_physics_for_scene(env_cfg, None)
    assert env_cfg.scene.replicate_physics is False


def _test_deformable_sphere_droid_newton_smoke(simulation_app) -> bool:
    """Boot the shipped deformable env with DROID on Newton VBD and check the soft body simulates."""
    import torch

    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena_environments.cli import (
        build_environment_from_cli,
        ensure_environments_registered,
        get_isaaclab_arena_environments_cli_parser,
    )

    ensure_environments_registered()
    factory_type = EnvironmentRegistry().get_component_by_name("deformable_sphere_pick_place")

    parser = get_isaaclab_arena_environments_cli_parser(get_isaaclab_arena_cli_parser())
    args_cli = parser.parse_args([
        "--num_envs",
        "1",
        "--presets",
        "newton_mjwarp_vbd",
        "deformable_sphere_pick_place",
        "--embodiment",
        "droid_abs_joint_pos",
    ])

    arena_env = build_environment_from_cli(factory_type, args_cli)
    builder = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli))
    env = builder.make_registered().unwrapped
    try:
        # Newton VBD backend was actually selected.
        assert env.cfg.scene.replicate_physics is True
        assert env.action_manager.total_action_dim == 8

        env.reset()
        asset = env.scene["procedural_deformable_sphere"]
        nodal_before = asset.data.nodal_pos_w.torch.clone()
        assert nodal_before.shape[1] > 0, "deformable has no simulation nodes"
        assert torch.isfinite(nodal_before).all(), "nodal positions not finite after reset"

        zero_action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        for _ in range(15):
            env.step(zero_action)

        nodal_after = asset.data.nodal_pos_w.torch
        assert torch.isfinite(nodal_after).all(), "nodal positions diverged (non-finite) after stepping"
        # The VBD solver must actually advance the soft body under gravity/contact.
        max_delta = (nodal_after - nodal_before).abs().max().item()
        assert max_delta > 1e-5, f"deformable did not move under Newton stepping (max delta {max_delta})"
    finally:
        env.close()
    return True


@pytest.mark.with_subprocess
def test_deformable_sphere_droid_newton_smoke() -> None:
    assert run_simulation_app_function(_test_deformable_sphere_droid_newton_smoke, headless=HEADLESS)
