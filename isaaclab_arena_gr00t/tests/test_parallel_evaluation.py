# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

from isaaclab_arena_gr00t.parallel_evaluation import (
    RTX_KIT_ARGS,
    TASK_BASE_SEEDS,
    _build_parser,
    _process_environment,
    _python_package_versions,
    _worker_command,
    build_worker_overrides,
    split_episode_budget,
    summarize_results,
)


def test_split_episode_budget_uses_all_eight_workers():
    episode_counts = split_episode_budget(total_episodes=10, worker_count=8)

    assert episode_counts == [2, 2, 1, 1, 1, 1, 1, 1]
    assert sum(episode_counts) == 10


def test_worker_overrides_use_one_env_and_distinct_eval_seeds():
    rank_zero_overrides = build_worker_overrides(rank=0, episode_count=2)
    rank_seven_overrides = build_worker_overrides(rank=7, episode_count=1)

    for task_name, base_seed in TASK_BASE_SEEDS.items():
        assert f"runs.{task_name}.environment_builder.num_envs=1" in rank_zero_overrides
        assert f"runs.{task_name}.environment_builder.seed={base_seed}" in rank_zero_overrides
        assert f"runs.{task_name}.policy.sensor_seed={base_seed}" in rank_zero_overrides
        assert f"runs.{task_name}.rollout_limit.num_episodes=2" in rank_zero_overrides
        assert f"runs.{task_name}.environment_builder.seed={base_seed + 7}" in rank_seven_overrides
        assert f"runs.{task_name}.policy.sensor_seed={base_seed + 7}" in rank_seven_overrides
        assert f"runs.{task_name}.rollout_limit.num_episodes=1" in rank_seven_overrides
        assert f"runs.{task_name}.environment.randomize_policy_start_pose=false" in rank_zero_overrides
        assert f"runs.{task_name}.environment.randomize_policy_start_pose=false" in rank_seven_overrides


def test_randomized_start_pose_requires_explicit_opt_in():
    overrides = build_worker_overrides(
        rank=0,
        episode_count=1,
        task_name="franka_blue_tray_1_cube",
        randomize_policy_start_pose=True,
    )

    assert "runs.franka_blue_tray_1_cube.environment.randomize_policy_start_pose=true" in overrides


def test_repeated_task_options_select_multiple_tasks():
    args = _build_parser().parse_args([
        "--task",
        "franka_blue_tray_1_cube",
        "--task",
        "franka_blue_tray_2_cubes",
    ])

    assert args.task == ["franka_blue_tray_1_cube", "franka_blue_tray_2_cubes"]


def test_single_task_worker_removes_other_runs_and_syncs_rtx(tmp_path):
    selected_task = "franka_blue_tray_2_cubes"
    overrides = build_worker_overrides(rank=3, episode_count=1, task_name=selected_task)

    assert f"runs.{selected_task}.environment_builder.seed=20010" in overrides
    assert "~runs.franka_blue_tray_1_cube" in overrides
    assert "~runs.franka_blue_tray_3_cubes" in overrides
    assert all("franka_blue_tray_1_cube.environment_builder" not in value for value in overrides)

    args = type(
        "Args",
        (),
        {
            "arena_python": tmp_path / "arena-python",
            "arena_repo": tmp_path / "arena",
            "experiment_config": tmp_path / "experiment.yaml",
            "record_camera_video": True,
            "randomize_policy_start_pose": False,
        },
    )()
    command = _worker_command(args, 3, 5658, 1, tmp_path / "output", selected_task)

    assert f"--kit_args={RTX_KIT_ARGS}" in command
    assert "--record_camera_video" in command
    assert "~runs.franka_blue_tray_1_cube" in command


def test_process_environment_uses_local_cosmos_model(tmp_path):
    environment = _process_environment(0, tmp_path / "gr00t", tmp_path / "cosmos", tmp_path / "arena")

    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert environment["GROOT_COSMOS_MODEL_PATH"] == str(tmp_path / "cosmos")
    assert environment["PYTHONPATH"].split(":")[:2] == [str(tmp_path / "arena"), str(tmp_path / "gr00t")]


def test_python_package_versions_prefers_container_version_files(tmp_path, monkeypatch):
    isaac_sim_root = tmp_path / "isaac-sim"
    isaac_lab_root = tmp_path / "isaaclab"
    isaac_sim_root.mkdir()
    isaac_lab_root.mkdir()
    (isaac_sim_root / "VERSION").write_text("6.0.1-test\n", encoding="utf-8")
    (isaac_lab_root / "VERSION").write_text("3.0.0-test\n", encoding="utf-8")
    monkeypatch.setenv("ISAAC_PATH", str(isaac_sim_root))
    monkeypatch.setenv("ISAAC_LAB_ROOT", str(isaac_lab_root))

    versions = _python_package_versions(
        Path(sys.executable),
        ("isaacsim", "isaaclab", "definitely-not-an-installed-package"),
    )

    assert versions == {
        "isaacsim": "6.0.1-test",
        "isaaclab": "3.0.0-test",
        "definitely-not-an-installed-package": "not-installed",
    }


def test_summarize_results_groups_worker_outputs_by_task(tmp_path):
    for rank in range(2):
        for task_name in TASK_BASE_SEEDS:
            results_dir = tmp_path / f"rank-{rank:02d}" / task_name
            results_dir.mkdir(parents=True)
            record = {
                "job_name": task_name,
                "env_id": 0,
                "episode_in_env": 0,
                "success": rank == 0,
            }
            (results_dir / "episode_results_rebuild0.jsonl").write_text(
                json.dumps(record) + "\n",
                encoding="utf-8",
            )

    summary = summarize_results(tmp_path, expected_episodes_per_task=2)

    for task_summary in summary.values():
        assert task_summary == {"episodes": 2, "successes": 1, "success_rate": 0.5}
