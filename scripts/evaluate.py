"""Evaluate a trained QSNE policy on the 2D simulator.

Reports the four headline metrics used in the paper's Table~2:
    - Success Rate  (% of trials that reach the goal)
    - Path Length   (mean Euclidean trajectory length, meters)
    - Time to Goal  (mean steps to goal, seconds at 10 Hz)
    - Collisions    (collisions per trial)

The script runs ten trials per scenario by default to match the protocol of
Section 3.2 of the paper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from qsne.aggregation import LLM_QUERY_INTERVAL
from qsne.env import QSNEEnv, QSNEEnvConfig, default_office_world, default_outdoor_world
from qsne.llm_module import AsyncLLMModule, build_prompt
from qsne.networks import LLM_EMBED_DIM
from qsne.policy import QSNEPolicy


WORLDS = {
    "office": default_office_world,
    "outdoor": default_outdoor_world,
}


def make_env(args):
    """Build the evaluation environment for the requested backend.

    Mirrors `make_env` in scripts/train.py. Both backends produce the same
    observation dict so the trial loop is backend-agnostic.
    """
    if args.backend == "sim2d":
        env_cfg = QSNEEnvConfig(
            world_factory=WORLDS[args.world], seed=args.seed,
        )
        return QSNEEnv(env_cfg)
    if args.backend == "gazebo":
        from qsne.gazebo_env import (
            GazeboEnvConfig, GazeboQSNEEnv, WORLD_PRESETS,
        )
        preset = WORLD_PRESETS.get(args.world, {})
        env_cfg = GazeboEnvConfig(seed=args.seed, **preset)
        return GazeboQSNEEnv(env_cfg)
    raise ValueError(f"Unknown backend: {args.backend}")


def run_trial(
    env: QSNEEnv,
    policy: QSNEPolicy,
    llm: AsyncLLMModule | None,
    max_steps: int = 1500,
) -> dict:
    """Run a single evaluation trial and return its metrics."""
    obs = env.reset()
    hidden = policy.net.init_hidden(batch_size=1)
    prev_pos = obs["odom"][:2].copy()

    path_length = 0.0
    collisions = 0
    reached_goal = False
    final_step = max_steps

    for step in range(max_steps):
        if llm is not None and obs["should_query_llm"]:
            llm.trigger(build_prompt(obs["sector_sum"], obs["odom"]))

        u_t = torch.from_numpy(obs["u"]).unsqueeze(0)
        e_t = (
            llm.get_embedding().unsqueeze(0) if llm is not None
            else torch.zeros(1, LLM_EMBED_DIM, dtype=torch.float32)
        )
        with torch.no_grad():
            out = policy.act(u_t, e_t, hidden, deterministic=True)
        action = out["action"].squeeze(0).numpy()
        hidden = out["hidden"]

        obs, reward, done, info = env.step(action)
        path_length += float(np.linalg.norm(obs["odom"][:2] - prev_pos))
        prev_pos = obs["odom"][:2].copy()
        if info["collided"]:
            collisions += 1
        if done:
            reached_goal = bool(info["reached_goal"])
            final_step = step + 1
            break

    return {
        "reached_goal": reached_goal,
        "path_length": path_length,
        "steps": final_step,
        "collisions": collisions,
    }


def summarize(trials: list[dict]) -> dict:
    """Aggregate per-trial metrics into the table format of the paper."""
    n = len(trials)
    successes = sum(1 for t in trials if t["reached_goal"])
    successful = [t for t in trials if t["reached_goal"]]
    path_lengths = np.array(
        [t["path_length"] for t in successful], dtype=np.float32
    ) if successful else np.array([0.0], dtype=np.float32)
    steps_arr = np.array(
        [t["steps"] for t in successful], dtype=np.float32
    ) if successful else np.array([0.0], dtype=np.float32)
    coll_arr = np.array([t["collisions"] for t in trials], dtype=np.float32)
    return {
        "success_rate_pct": 100.0 * successes / n,
        "path_length_mean": float(path_lengths.mean()),
        "path_length_std": float(path_lengths.std()),
        "time_to_goal_s_mean": float(0.1 * steps_arr.mean()),  # 10 Hz -> 0.1 s/step
        "time_to_goal_s_std": float(0.1 * steps_arr.std()),
        "collisions_mean": float(coll_arr.mean()),
        "collisions_std": float(coll_arr.std()),
        "num_trials": n,
        "num_successes": successes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a QSNE checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--world", choices=list(WORLDS.keys()), default="office")
    parser.add_argument(
        "--backend", choices=["sim2d", "gazebo"], default="sim2d",
        help=(
            "Evaluation backend. 'sim2d' uses the lightweight polygonal "
            "simulator in qsne/env.py. 'gazebo' uses the gym-gazebo style "
            "wrapper in qsne/gazebo_env.py and requires Gazebo + Husky to "
            "be already running."
        ),
    )
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--max-steps", type=int, default=1500)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    policy = QSNEPolicy()
    policy.load_state_dict(ckpt["policy_state"])
    policy.eval()

    env = make_env(args)
    llm = None if args.no_llm else AsyncLLMModule()

    trials = []
    for i in range(args.num_trials):
        # New seed per trial so that the goal/start are different each time.
        env.cfg.seed = args.seed + i
        env.rng = np.random.default_rng(env.cfg.seed)
        result = run_trial(env, policy, llm, args.max_steps)
        trials.append(result)
        print(
            f"Trial {i + 1:>2d}/{args.num_trials}: "
            f"success={result['reached_goal']} "
            f"steps={result['steps']:>4d} "
            f"path={result['path_length']:.2f} m "
            f"collisions={result['collisions']}"
        )

    summary = summarize(trials)
    print()
    print(f"=== QSNE evaluation summary ({args.world}, {args.num_trials} trials) ===")
    print(f"Success Rate (%) : {summary['success_rate_pct']:.1f}")
    print(
        f"Path Length (m)  : "
        f"{summary['path_length_mean']:.2f} ± {summary['path_length_std']:.2f}"
    )
    print(
        f"Time to Goal (s) : "
        f"{summary['time_to_goal_s_mean']:.1f} ± {summary['time_to_goal_s_std']:.1f}"
    )
    print(
        f"Collisions/Trial : "
        f"{summary['collisions_mean']:.2f} ± {summary['collisions_std']:.2f}"
    )

    if llm is not None:
        llm.shutdown()


if __name__ == "__main__":
    main()
