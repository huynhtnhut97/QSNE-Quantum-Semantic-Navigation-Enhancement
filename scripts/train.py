"""Train the QSNE policy in the lightweight 2D simulator.

Usage
-----
    python scripts/train.py --total-steps 100000 --seed 0 \
        --save-path checkpoints/qsne_seed0.pt

The training protocol matches Section 2.3 of the paper:
    - Total time steps                : 1e5
    - Learning rate                   : 3e-4
    - Batch size                      : 64
    - Entropy coefficient             : 1e-2
    - Clip parameter (epsilon)        : 0.2
    - Discount factor (gamma)         : 0.99
    - GAE lambda                      : 0.95

The script trains a single seed end-to-end and writes a checkpoint to disk
on completion. To reproduce the five-seed ablation reported in the paper,
launch five copies of this script with seeds [0, 1, 2, 3, 4].
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from qsne.aggregation import LLM_QUERY_INTERVAL
from qsne.buffer import RolloutBuffer
from qsne.env import QSNEEnv, QSNEEnvConfig, default_office_world, default_outdoor_world
from qsne.llm_module import AsyncLLMModule, build_prompt
from qsne.networks import LLM_EMBED_DIM, PQC_FEATURE_DIM, SCAN_DIM
from qsne.policy import QSNEPolicy
from qsne.ppo import PPOConfig, PPOTrainer


WORLDS = {
    "office": default_office_world,
    "outdoor": default_outdoor_world,
}


def make_env(args):
    """Build the training environment for the requested backend.

    The two backends produce the same observation dict so the rest of the
    training loop is backend-agnostic.
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


def set_global_seed(seed: int) -> None:
    """Propagate the seed to numpy, torch, and python's random module."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_rollout(
    env: QSNEEnv,
    policy: QSNEPolicy,
    buffer: RolloutBuffer,
    llm: AsyncLLMModule | None,
    rollout_steps: int,
) -> dict:
    """Collect `rollout_steps` transitions and store them in the buffer.

    Returns aggregate episode statistics for logging.
    """
    buffer.reset()
    obs = env.reset()
    hidden = policy.net.init_hidden(batch_size=1)

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    current_return = 0.0
    current_length = 0
    n_collected = 0

    while n_collected < rollout_steps:
        # Trigger an LLM call when appropriate.
        if llm is not None and obs["should_query_llm"]:
            llm.trigger(build_prompt(obs["sector_sum"], obs["odom"]))

        u_t = torch.from_numpy(obs["u"]).unsqueeze(0)
        if llm is not None:
            e_t = llm.get_embedding().unsqueeze(0)
        else:
            e_t = torch.zeros(1, LLM_EMBED_DIM, dtype=torch.float32)

        with torch.no_grad():
            out = policy.act(u_t, e_t, hidden, deterministic=False)
        action = out["action"].squeeze(0).numpy()
        log_prob = float(out["log_prob"].item())
        value = float(out["value"].item())

        next_obs, reward, done, info = env.step(action)

        buffer.add(
            obs=obs["u"],
            embed=e_t.squeeze(0).numpy(),
            clean_scan=obs["clean_scan"],
            action=action,
            log_prob=log_prob,
            reward=reward,
            value=value,
            done=done,
        )
        hidden = out["hidden"]
        current_return += reward
        current_length += 1
        n_collected += 1

        if done:
            episode_returns.append(current_return)
            episode_lengths.append(current_length)
            current_return = 0.0
            current_length = 0
            obs = env.reset()
            hidden = policy.net.init_hidden(batch_size=1)
        else:
            obs = next_obs

    # Compute the bootstrap value for GAE on the last non-terminal state.
    if not done:
        u_t = torch.from_numpy(obs["u"]).unsqueeze(0)
        e_t = (
            llm.get_embedding().unsqueeze(0) if llm is not None
            else torch.zeros(1, LLM_EMBED_DIM, dtype=torch.float32)
        )
        with torch.no_grad():
            last_value = float(
                policy.forward(u_t, e_t, hidden)["value"].squeeze().item()
            )
    else:
        last_value = 0.0
    buffer.finalize(last_value=last_value)

    return {
        "mean_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "mean_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "num_episodes": len(episode_returns),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train QSNE in the 2D simulator.")
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--rollout-steps", type=int, default=2_048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--world", choices=list(WORLDS.keys()), default="office")
    parser.add_argument(
        "--backend", choices=["sim2d", "gazebo"], default="sim2d",
        help=(
            "Training environment backend. 'sim2d' uses the lightweight "
            "polygonal simulator in qsne/env.py. 'gazebo' uses the "
            "gym-gazebo style wrapper in qsne/gazebo_env.py and requires "
            "Gazebo + Husky to be already running (see qsne_training.launch)."
        ),
    )
    parser.add_argument("--save-path", type=str, default="checkpoints/qsne.pt")
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Disable the LLM module; zero embedding only (PQC-only ablation).",
    )
    parser.add_argument(
        "--log-interval", type=int, default=1,
        help="Print one log line per N PPO updates.",
    )
    args = parser.parse_args()

    set_global_seed(args.seed)

    env = make_env(args)
    policy = QSNEPolicy()
    trainer = PPOTrainer(policy, PPOConfig())
    buffer = RolloutBuffer(
        capacity=args.rollout_steps,
        obs_dim=PQC_FEATURE_DIM,
        embed_dim=LLM_EMBED_DIM,
        scan_dim=SCAN_DIM,
        gamma=trainer.cfg.gamma,
        gae_lambda=trainer.cfg.gae_lambda,
    )
    llm = None if args.no_llm else AsyncLLMModule()

    os.makedirs(Path(args.save_path).parent, exist_ok=True)

    start = time.time()
    total_collected = 0
    update_idx = 0
    while total_collected < args.total_steps:
        rollout_stats = collect_rollout(
            env, policy, buffer, llm, args.rollout_steps
        )
        update_stats = trainer.update(buffer)
        total_collected += args.rollout_steps
        update_idx += 1

        if update_idx % args.log_interval == 0:
            elapsed = time.time() - start
            print(
                f"[step {total_collected:>7d}] "
                f"return={rollout_stats['mean_return']:.2f} "
                f"len={rollout_stats['mean_length']:.1f} "
                f"loss={update_stats['loss_total']:.3f} "
                f"pi={update_stats['loss_policy']:.3f} "
                f"v={update_stats['loss_value']:.3f} "
                f"scan={update_stats['loss_scan']:.3f} "
                f"kl={update_stats['kl']:.4f} "
                f"elapsed={elapsed:.1f}s"
            )

    torch.save(
        {
            "policy_state": policy.state_dict(),
            "ppo_config": trainer.cfg.__dict__,
            "args": vars(args),
        },
        args.save_path,
    )
    print(f"Saved checkpoint to {args.save_path}")

    if llm is not None:
        llm.shutdown()


if __name__ == "__main__":
    main()
