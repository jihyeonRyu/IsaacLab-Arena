# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json

from isaaclab_arena_gr00t.parallel_evaluation import (
    TASK_BASE_SEEDS,
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
        assert f"runs.{task_name}.rollout_limit.num_episodes=2" in rank_zero_overrides
        assert f"runs.{task_name}.environment_builder.seed={base_seed + 7}" in rank_seven_overrides
        assert f"runs.{task_name}.rollout_limit.num_episodes=1" in rank_seven_overrides


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
