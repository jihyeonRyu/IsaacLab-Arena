# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg
from isaaclab.sim import RenderCfg, SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_contrib.deformable.newton_manager_cfg import CoupledMJWarpVBDSolverCfg, NewtonModelCfg, VBDSolverCfg

# Import from the package root so this resolves whether MJWarpSolverCfg lives in
# newton_manager_cfg (older isaaclab_newton) or mjwarp_manager_cfg (Isaac Lab Beta 2).
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import PresetCfg


@configclass
class DeformableNewtonCfg(NewtonCfg):
    """Newton physics config with global deformable-object model parameters."""

    model_cfg: NewtonModelCfg | None = None
    """Global Newton model parameters applied after builder finalization."""


@configclass
class ArenaPhysicsCfg(PresetCfg):
    """Physics backend presets available to all Arena environments.

    ``default`` / ``physx`` use the stock PhysX backend.
    ``newton`` uses MuJoCo-Warp via Newton with solver parameters tuned
    for dexterous manipulation (matches ``KukaAllegroPhysicsCfg.newton``).
    ``newton_mjwarp_vbd`` couples the Newton rigid solver with the VBD soft-body solver.
    """

    physx = PhysxCfg()
    newton = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=300,
            nconmax=400,
            impratio=10.0,
            cone="elliptic",
            update_data_interval=2,
            iterations=100,
            ls_iterations=15,
            ls_parallel=False,
            use_mujoco_contacts=False,
            ccd_iterations=15000,
        ),
        num_substeps=2,
        debug_mode=False,
    )
    newton_mjwarp_vbd = DeformableNewtonCfg(
        solver_cfg=CoupledMJWarpVBDSolverCfg(
            # Rigid solver settings favor contact-rich manipulation: a high contact budget
            # (njmax/nconmax), elliptic friction cone with large impratio for stable grasps, and many
            # solver/CCD iterations so gripper-object contacts resolve without penetration.
            rigid_solver_cfg=MJWarpSolverCfg(
                solver="newton",
                integrator="implicitfast",
                njmax=300,
                nconmax=400,
                impratio=10.0,
                cone="elliptic",
                update_data_interval=2,
                iterations=100,
                ls_iterations=15,
                ls_parallel=False,
                ccd_iterations=15000,
            ),
            # Self-contact and periodic collision re-detection are off: the objects are convex and
            # small, so the extra cost buys nothing.
            soft_solver_cfg=VBDSolverCfg(
                iterations=10,
                integrate_with_external_rigid_solver=True,
                particle_enable_self_contact=False,
                particle_collision_detection_interval=-1,
            ),
            coupling_mode="two_way",
        ),
        # Contact stiffness (ke), damping (kd), and friction (mu) for soft contacts (deformable) and
        # rigid shape contacts. shape_material_kd damps rigid-body contact: without it, a rigid object
        # resting on the table bounces on its penetration and the high friction turns that bounce into
        # lateral skitter, so it walks off the table; kd=100 lets rigid objects settle and stay put.
        model_cfg=NewtonModelCfg(
            soft_contact_ke=1.0e4,
            soft_contact_kd=1.0e-5,
            soft_contact_mu=5.0,
            shape_material_ke=4.0e4,
            shape_material_kd=100.0,
            shape_material_mu=5.0,
        ),
        num_substeps=10,
        debug_mode=False,
    )
    default = physx


@configclass
class IsaacLabArenaManagerBasedRLEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for an IsaacLab Arena environment."""

    # NOTE(alexmillane, 2025-07-29): The following definitions are taken from the base class.
    # scene: InteractiveSceneCfg
    # observations: object
    # actions: object
    # events: object
    # terminations: object
    # recorders: object

    # Kill the unused managers
    commands = None
    rewards = None
    curriculum = None

    metrics: object | None = None

    episode_recorders: object | None = None

    # Task language description
    task_description: str | None = None

    # Override the RTX renderer's built-in scene ambient (carb /rtx/sceneDb/ambientLightIntensity, default 1.0 with
    # color [0.1, 0.1, 0.1]) so that USD light prims fully control scene illumination.
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=2,
        render=RenderCfg(
            carb_settings={
                "/rtx/sceneDb/ambientLightIntensity": 0.0,
                # Workaround for IsaacLab #6424: stop the physx-tensors filter matcher from
                # recursing into leaf collision shapes so a contact filter pointing at a rigid
                # body with multiple collision shapes resolves to a single entry (otherwise the
                # view fails with "expected 1, found N").
                "/physics/tensors/recursiveLeafPatternMatch": False,
            },
        ),
    )
    decimation: int = 4
    wait_for_textures: bool = False


@configclass
class IsaacArenaManagerBasedMimicEnvCfg(IsaacLabArenaManagerBasedRLEnvCfg, MimicEnvCfg):
    """Configuration for an IsaacLab Arena environment."""

    # NOTE(alexmillane, 2025-09-10): The following members are defined in the MimicEnvCfg class.
    # Restated here for clarity.
    # datagen_config: DataGenConfig = DataGenConfig()
    # subtask_configs: dict[str, list[SubTaskConfig]] = {}
    # task_constraint_configs: list[SubTaskConstraintConfig] = []

    # Data generation keeps the longer historical default so demos are not truncated; the task's
    # (shorter) episode length is only applied to non-mimic RL/eval envs by the env builder.
    episode_length_s: float = 50.0
