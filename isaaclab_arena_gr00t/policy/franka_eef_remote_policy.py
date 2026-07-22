# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GR00T remote adapter for the Franka EEF-relative LeRobot dataset."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from dataclasses import dataclass
from typing import Any

from gr00t.policy.server_client import PolicyClient as Gr00tPolicyClient

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _source_compatible_rot6d(quaternion: torch.Tensor) -> torch.Tensor:
    """Reproduce the converter's historical raw-quaternion-as-xyzw convention."""
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1.0e-8)
    x, y, z, w = quaternion.unbind(dim=-1)
    row0 = torch.stack(
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)), dim=-1
    )
    row1 = torch.stack(
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)), dim=-1
    )
    return torch.cat((row0, row1), dim=-1)


def _find_action(output: dict[str, Any], key: str) -> np.ndarray:
    for candidate in (key, f"action.{key}"):
        if candidate in output:
            return np.asarray(output[candidate])
    nested = output.get("action")
    if isinstance(nested, dict) and key in nested:
        return np.asarray(nested[key])
    raise KeyError(f"GR00T response does not contain action key '{key}'; available={sorted(output)}")


@dataclass
class FrankaEefRemotePolicyCfg(PolicyCfg):
    """Configure the remote Franka GR00T policy and its rate adapter."""

    num_envs: int = 1
    remote_host: str = "localhost"
    remote_port: int = 5555
    remote_api_token: str | None = None
    policy_device: str = "cuda:0"
    action_horizon: int = 16
    action_chunk_length: int = 16
    policy_hz: int = 15
    control_hz: int = 60
    interpolate_actions: bool = True
    external_camera_name: str = "external_camera_rgb"
    wrist_camera_name: str = "wrist_camera_rgb"
    external_video_key: str = "external"
    wrist_video_key: str = "wrist"
    eef_state_key: str = "eef_pose"
    gripper_state_key: str = "gripper"
    eef_action_key: str = "eef_delta"
    gripper_action_key: str = "gripper"
    language_key: str = "annotation.human.action.task_description"
    max_translation_action: float = 0.08
    max_rotation_action: float = 0.15

    def __post_init__(self) -> None:
        assert self.num_envs > 0
        assert 0 < self.action_chunk_length <= self.action_horizon
        assert self.control_hz >= self.policy_hz and self.control_hz % self.policy_hz == 0


@register_policy
class FrankaEefRemotePolicy(PolicyBase[FrankaEefRemotePolicyCfg]):
    """Translate Arena observations to our GR00T modality and return 7D IK-relative actions."""

    name = "franka_eef_gr00t_remote"

    def __init__(self, config: FrankaEefRemotePolicyCfg):
        super().__init__(config)
        self.device = torch.device(config.policy_device)
        self.repeat_factor = config.control_hz // config.policy_hz
        self.task_description: str | None = None
        self._client: Gr00tPolicyClient | None = Gr00tPolicyClient(
            host=config.remote_host,
            port=config.remote_port,
            api_token=config.remote_api_token,
            strict=False,
        )
        if not self._client.ping():
            raise ConnectionError(f"Cannot reach GR00T server at {config.remote_host}:{config.remote_port}")

        self._chunk = torch.zeros(
            (config.num_envs, config.action_horizon, 7), dtype=torch.float32, device=self.device
        )
        self._policy_index = torch.full(
            (config.num_envs,), config.action_chunk_length, dtype=torch.long, device=self.device
        )
        self._control_substep = torch.zeros(config.num_envs, dtype=torch.long, device=self.device)


    def set_task_description(self, task_description: str | None) -> str:
        if not task_description:
            raise ValueError("The Franka evaluation task must provide a language instruction")
        self.task_description = task_description
        return task_description

    def _build_request(self, observation: dict[str, Any]) -> dict[str, Any]:
        cfg = self.config
        camera_obs = observation["camera_obs"]
        external = _to_numpy(camera_obs[cfg.external_camera_name])
        wrist = _to_numpy(camera_obs[cfg.wrist_camera_name])
        if external.ndim != 4 or wrist.ndim != 4:
            raise ValueError(f"Expected batched NHWC RGB; got external={external.shape}, wrist={wrist.shape}")

        policy_obs = observation["policy"]
        eef_pos = policy_obs["eef_pos"].to(device=self.device, dtype=torch.float32)
        eef_quat = policy_obs["eef_quat"].to(device=self.device, dtype=torch.float32)
        gripper_obs = policy_obs["gripper_pos"].to(device=self.device, dtype=torch.float32)
        # FrankaObservationsCfg stores [finger_2, -finger_1]. Their difference is total opening width.
        gripper_width = (gripper_obs[:, :1] - gripper_obs[:, 1:2]).clamp_min(0.0)
        eef_pose = torch.cat((eef_pos, _source_compatible_rot6d(eef_quat)), dim=-1)

        assert self.task_description is not None
        num_envs = external.shape[0]
        return {
            "language": {cfg.language_key: [[self.task_description] for _ in range(num_envs)]},
            "video": {
                cfg.external_video_key: external[:, None, ...],
                cfg.wrist_video_key: wrist[:, None, ...],
            },
            "state": {
                cfg.eef_state_key: _to_numpy(eef_pose)[:, None, :],
                cfg.gripper_state_key: _to_numpy(gripper_width)[:, None, :],
            },
        }

    def _fetch_chunk(self, observation: dict[str, Any]) -> torch.Tensor:
        assert self._client is not None, "GR00T remote policy is closed"
        response = self._client.get_action(self._build_request(observation))
        action_output = response[0] if isinstance(response, tuple) else response
        eef = _find_action(action_output, self.config.eef_action_key)
        gripper = _find_action(action_output, self.config.gripper_action_key)
        if eef.ndim != 3 or eef.shape[-1] != 6:
            raise ValueError(f"Expected EEF action shape (N,H,6), got {eef.shape}")
        if gripper.ndim == 2:
            gripper = gripper[..., None]
        if gripper.ndim != 3 or gripper.shape[-1] != 1:
            raise ValueError(f"Expected gripper action shape (N,H,1), got {gripper.shape}")
        actions = torch.as_tensor(np.concatenate((eef, gripper), axis=-1), device=self.device, dtype=torch.float32)
        if actions.shape[0] != self.config.num_envs or actions.shape[1] < self.config.action_horizon:
            raise ValueError(
                f"Expected action chunk ({self.config.num_envs},>={self.config.action_horizon},7), got {actions.shape}"
            )
        actions[:, :, :3].clamp_(-self.config.max_translation_action, self.config.max_translation_action)
        actions[:, :, 3:6].clamp_(-self.config.max_rotation_action, self.config.max_rotation_action)
        actions[:, :, 6].clamp_(-1.0, 1.0)
        return actions[:, : self.config.action_horizon]

    def get_action(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        needs_chunk = self._policy_index >= self.config.action_chunk_length
        if needs_chunk.any():
            fetched = self._fetch_chunk(observation)
            self._chunk[needs_chunk] = fetched[needs_chunk]
            self._policy_index[needs_chunk] = 0
            self._control_substep[needs_chunk] = 0

        batch = torch.arange(self.config.num_envs, device=self.device)
        current = self._chunk[batch, self._policy_index]
        action = current.clone()
        if self.config.interpolate_actions and self.repeat_factor > 1:
            next_index = torch.minimum(
                self._policy_index + 1,
                torch.full_like(self._policy_index, self.config.action_chunk_length - 1),
            )
            following = self._chunk[batch, next_index]
            alpha = (self._control_substep.float() / self.repeat_factor).unsqueeze(-1)
            action[:, :6] = torch.lerp(current[:, :6], following[:, :6], alpha)
        # Gripper commands are categorical; hold instead of interpolating through zero.
        action[:, 6] = current[:, 6]

        self._control_substep += 1
        advance = self._control_substep >= self.repeat_factor
        self._policy_index[advance] += 1
        self._control_substep[advance] = 0
        return action.to(device=env.unwrapped.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.config.num_envs, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device)
        self._chunk[env_ids] = 0.0
        self._policy_index[env_ids] = self.config.action_chunk_length
        self._control_substep[env_ids] = 0
        if self._client is not None:
            self._client.reset()

    def shutdown_remote(self, kill_server: bool = False) -> None:
        """Close the client. The externally launched GR00T server is never killed here."""
        del kill_server
        self.close()

    def close(self) -> None:
        client = self._client
        try:
            if client is not None:
                socket = getattr(client, "socket", None)
                context = getattr(client, "context", None)
                if socket is not None:
                    socket.close(linger=0)
                if context is not None:
                    context.term()
        finally:
            self._client = None
