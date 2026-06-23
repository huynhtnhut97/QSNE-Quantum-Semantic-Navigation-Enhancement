# QSNE — Quantum-Semantic Navigation Enhancement

Reference implementation of the QSNE framework from
*"A Learning Framework for Robust Navigation of Mobile Robots under Partial Observability"*
(Huynh, Sivak, Gutierrez, Nguyen).

QSNE addresses ground-robot navigation under severe LiDAR degradation
(per-ray dropout + Gaussian noise) on the Clearpath Husky UGV. The framework
combines four ideas in a single end-to-end pipeline:

1. **Observation aggregation.** A 720-ray scan is reduced to a 6-D normalized
   PQC input through 12-sector trimmed means and a polar projection of the
   goal-relative odometry.
2. **PQC encoder.** A 6-qubit, 4-layer parameterized quantum circuit
   (PennyLane, statevector simulation) lifts the 6-D input to a 6-D feature
   vector ⟨Z_i⟩ via R_X, R_Y rotations and a CNOT entangling chain.
3. **LLM semantic reasoning.** A GPT-4o call (asynchronous, off the control
   loop) produces an obstacle-and-action description that DistilBERT embeds
   into 256-D. The call fires every 10 steps or whenever a sector variance
   exceeds τ = 2.0.
4. **PPO-LSTM dual-head policy.** A 128-unit LSTM ingests the concatenated
   262-D vector and feeds two parallel heads: a Gaussian policy/value head
   on `/cmd_vel` and a 720-D scan-reconstruction decoder on `/scan_corrected`
   (consumed by Gmapping).

Two training backends are supported. The `GazeboQSNEEnv` backend matches
the paper's experimental setup (ROS Noetic + Gazebo + the Clearpath Husky
worlds) and is the primary training path. A lightweight 2-D polygonal
simulator is also included for laptop-scale development and unit tests.
Both backends expose the same observation dict, so the rest of the
pipeline is backend-agnostic. A separate ROS bridge handles deployment
of a trained policy on the physical Husky.

---

## Repository layout

```
qsne_repo/
├── qsne/
│   ├── aggregation.py    # 12-sector trimmed-mean + polar projection (Section 2.1)
│   ├── degradation.py    # Per-ray dropout + Gaussian noise (Section 2 obs model)
│   ├── pqc.py            # 6-qubit, 4-layer PennyLane PQC (Section 2.3)
│   ├── networks.py       # LSTM + two heads (Section 2 architecture)
│   ├── policy.py         # QSNEPolicy: PQC + LSTM + heads wrapper
│   ├── llm_module.py     # Async GPT-4o + DistilBERT encoder (Section 2.4)
│   ├── encoder.py        # Re-export of the DistilBERT encoder
│   ├── reward.py         # Reward function (Section 2 reward block)
│   ├── buffer.py         # GAE rollout buffer for PPO
│   ├── ppo.py            # PPO update loop with the scan-reconstruction loss
│   ├── env.py            # Lightweight 2-D polygonal simulator (laptop testing)
│   ├── gazebo_env.py     # gym-gazebo style wrapper (Gazebo + Husky training)
│   └── ros_bridge.py     # ROS Noetic bridge for deployment on the Husky
├── scripts/
│   ├── train.py          # PPO training entry point (--backend gazebo|sim2d)
│   ├── evaluate.py       # 10-trial evaluation script (--backend gazebo|sim2d)
│   └── ros_bridge_node.py# ROS node entry point (used by qsne.launch)
├── launch/
│   ├── qsne.launch           # Husky + Gmapping + move_base (deployment)
│   └── qsne_training.launch  # Gazebo + Husky (training)
├── config/
│   └── default.yaml      # Consolidated hyperparameters
├── requirements.txt
├── setup.py
└── README.md
```

---

## Installation

Python 3.8+ is required. ROS Noetic is only needed for hardware deployment.

```bash
# Clone, create a virtualenv, and install Python dependencies.
git clone <this-repo>
cd qsne_repo
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt

# Optional: enable the OpenAI path of the LLM module.
export OPENAI_API_KEY="sk-..."
```

For ROS deployment, install ROS Noetic separately and source the workspace
before launching `qsne.launch`:

```bash
source /opt/ros/noetic/setup.bash
```

---

## Quick start — train and evaluate

Both `train.py` and `evaluate.py` take a `--backend` flag that selects
the environment. The two backends expose the same observation dict, so
the rest of the pipeline is identical.

### Backend 1: Gazebo (matches the paper's experimental setup)

Bring up Gazebo + Husky in one terminal, then launch training in another:

```bash
# Terminal 1: Gazebo + the chosen Clearpath world, with the Husky spawned and paused.
roslaunch qsne qsne_training.launch world:=office_world.world

# Terminal 2: train QSNE against the running simulation.
python scripts/train.py \
    --backend gazebo \
    --world office \
    --total-steps 100000 \
    --seed 0 \
    --save-path checkpoints/qsne_office_seed0.pt
```

The Gazebo env drives `/gazebo/{unpause,pause}_physics` for synchronous
stepping, so PPO observes deterministic transitions per step rather than
race conditions against the simulator clock. The four world presets
(`office`, `construction`, `agriculture`, `inspection`) live in
`qsne/gazebo_env.py::WORLD_PRESETS` and only set the bounding box for
free start/goal sampling; refine `obstacles` in `GazeboEnvConfig` to
match the obstacle layout of your specific world. Topic names, model
name, and bumper configuration are all overridable through the same
config.

### Backend 2: 2-D polygonal simulator (laptop development)

No ROS, no Gazebo — useful for fast iteration on the algorithm code,
unit tests, and sanity checks:

```bash
python scripts/train.py \
    --backend sim2d \
    --world office \
    --total-steps 100000 \
    --seed 0 \
    --save-path checkpoints/qsne_seed0.pt
```

### Evaluation

The evaluation protocol matches Section 3.2 of the paper (10 trials per
condition):

```bash
python scripts/evaluate.py \
    --backend gazebo \
    --checkpoint checkpoints/qsne_office_seed0.pt \
    --world office \
    --num-trials 10
```

Swap `--backend gazebo` for `--backend sim2d` to evaluate against the
2-D simulator instead.

### Five-seed campaign

The five-seed campaign of the paper corresponds to five copies of
`train.py` with seeds `0..4`. Each run takes roughly 2 hours on an A100
GPU node and roughly 8 hours on a workstation CPU under the 2-D
backend. The Gazebo backend is slower per step but matches the paper's
exact protocol.

### Ablation variants

To reproduce the four ablation variants:

| Variant            | Training command flag    |
|--------------------|--------------------------|
| Full QSNE          | (default)                |
| PPO-LSTM + PQC     | `--no-llm`               |
| PPO-LSTM + LLM     | edit `policy.py` to skip the PQC |
| PPO-LSTM baseline  | both of the above        |

The two-line variants (PPO-LSTM + LLM and the baseline) are intentional:
each ablation is one short edit to `QSNEPolicy._features`.

---

## Deployment on the Husky UGV

```bash
roslaunch qsne qsne.launch \
    checkpoint:=/abs/path/to/qsne_seed0.pt \
    control_rate_hz:=10 \
    use_llm:=true
```

The launch file:

* Starts the QSNE bridge node, which subscribes to `/noisy-scan`, `/odom`,
  and `/move_base_simple/goal`, and publishes `/cmd_vel` and
  `/scan_corrected`.
* Starts `slam_gmapping` configured to consume `/scan_corrected` rather
  than `/scan`, which is the corrected-mapping path of Section 2.
* Starts `move_base` with the default global planner; the local planner is
  bypassed because the QSNE bridge publishes `/cmd_vel` directly.

Publish a goal in RViz with the "2D Nav Goal" tool, or programmatically:

```python
rostopic pub /move_base_simple/goal geometry_msgs/PoseStamped ...
```

---

## Hyperparameters

All hyperparameters live in `config/default.yaml` and are pinned to the
values reported in the paper's consolidated table. Edit the YAML or pass
overrides on the command line. The most important values:

| Block      | Hyperparameter                | Value     |
|------------|-------------------------------|-----------|
| PPO        | Total time steps              | 1e5       |
| PPO        | Learning rate                 | 3e-4      |
| PPO        | Batch size                    | 64        |
| PPO        | Entropy coefficient           | 1e-2      |
| PPO        | Clip parameter ε              | 0.2       |
| PPO        | Discount factor γ             | 0.99      |
| PPO        | GAE λ                         | 0.95      |
| LSTM       | Hidden units                  | 128       |
| LSTM       | Policy/value FC layers        | 2 × 64    |
| PQC        | Number of qubits              | 6         |
| PQC        | Number of variational layers  | 4         |
| PQC        | Rotation gates                | R_X, R_Y  |
| PQC        | Entangling gate               | CNOT      |
| LLM        | Model snapshot                | gpt-4o-2024-08-06 |
| LLM        | Embedding dimension           | 256       |
| LLM        | Query interval                | 10 steps  |
| LLM        | Variance trigger threshold    | 2.0       |
| Action     | v ∈ [0, 2.0] m/s              |           |
| Action     | ω ∈ [-1.0, 1.0] rad/s         |           |

---

## Notes on the LLM path

The LLM module is asynchronous by design (Section 2.4). The control loop
never blocks on the GPT-4o round trip (~620 ms median). When the API key
is missing or the network is unreachable, the module falls back to:

1. A local Llama-3.1-8B-Instruct model served by Ollama at
   `http://localhost:11434`, if available.
2. A deterministic rule-based response derived from the sector means.

Both fallbacks preserve the data flow into the transformer encoder, so the
policy continues to receive a structured 256-D embedding rather than
zeros. The fallback path is only invoked when the primary call fails.

---

## Citation

If this implementation is useful in your work, please cite the QSNE paper.
A BibTeX entry will appear here once the journal publication is finalized.

---

## License

MIT.
