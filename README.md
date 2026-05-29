# SwiftSage + Adaptive Memory & Self-Reflection

This repository extends [**SwiftSage**](https://arxiv.org/abs/2305.17390) — a dual-process agent for complex interactive reasoning — with two modules developed in this thesis work:

- **AMM (Adaptive Memory Module)** — episodic memory that is written online and retrieved to guide Swift retries and Sage planning.
- **SRM (Self-Reflection Module)** — pre-execution action gating plus a stagnation-triggered **Critic** that intervenes when the agent loops without progress.

The implementation keeps the original SwiftSage **Swift** (fast, imitation-learned policy) and **Sage** (slow, LLM-based planner) backbone. AMM and SRM plug into the existing escalation path in `eval_agent_fast_slow.py` / `eval_utils.py`.

---

## Relation to SwiftSage

[SwiftSage](https://arxiv.org/abs/2305.17390) (Lin et al., 2023) combines:

| Module | Role | In this repo |
|--------|------|----------------|
| **Swift** | Small encoder–decoder LM (Flan-T5) trained on oracle trajectories | `yuchenlin/swift_sw`, `get_model_output()`, top-1 validation |
| **Sage** | LLM subgoal planning + action grounding | Qwen2.5 via [vLLM](https://github.com/vllm-project/vllm) (`llm/qwen_vllm_client.py`) |

Original resources: [project page](https://yuchenlin.xyz/swiftsage/), [Hugging Face Swift checkpoint](https://huggingface.co/yuchenlin/swift_sw), [ScienceWorld](https://sciworld.apps.allenai.org).

**What we add:** memory-augmented recovery before/full Sage calls (AMM), and reflection layers that validate actions and break stagnation (SRM).


### Per-step control flow (simplified)

1. **Buffer drain** — If Sage or Critic queued actions, execute head through SRM Gate (with optional deferred Critic after buffer empties).
2. **Swift** — Compose prompt (`compose_instance_v4`), run Flan-T5, validate top-1 action.
3. **T1 ladder (AMM)** — On Swift failure: retrieve success-only EMs (S1) → retry Swift with memory block; then success+near-miss (S2) → retry; else escalate.
4. **Sage (T4 + planning)** — Retrieve episodic memories for planning (S2; optional avoidance **B** if recent failures); inject into planning prompt; generate plan + short action sequence into buffer.
5. **SRM Gate** — Every executed action (Swift, buffer, repaired) passes parse/validity/no-op checks before `env.step`.
6. **Post-step** — Update stagnation detector; on stagnation, optionally run **Critic** (with optional AMM episodic block) to refill buffer.
7. **AMM write** — Classify step (success / near-miss / avoidance) and append structured EM to Letta when enabled.

---

## Repository layout

| Path | Description |
|------|-------------|
| `eval_agent_fast_slow.py` | Main evaluation loop (SwiftSage + AMM + SRM hooks) |
| `eval_utils.py` | Action selection, Sage prompts, T1/T4 retrieval orchestration |
| `run_eval_fast_slow.sh` | Batch runner with ablation profiles |
| `amm/` | Adaptive Memory Module (write, retrieve, formatters, Letta client) |
| `srm/` | Self-Reflection Module (gate, validator, repairer, stagnation, critic) |
| `llm/` | Qwen vLLM client for Sage and Critic |
| `data_utils/` | Prompt composition (`compose_instance_v4`), demos, preprocessing |
| `fast_agent/` | Swift training scripts (optional; pretrained checkpoint available) |
| `baselines/` | SayCan, ReAct, Reflexion, fast-only baselines |
| `analysis/` | Score aggregation and tables from run logs |

---

## Adaptive Memory Module (AMM)

**Backend:** [Letta](https://github.com/letta-ai/letta) cloud API (`LETTA_API_TOKEN`, `LETTA_AGENT_ID`).

**Memory types** (auto-tagged from reward / observation):

- **Success** — meaningful positive progress  
- **Near-miss** — partial progress without full milestone  
- **Avoidance** — invalid or blocked actions (TTL in config)

**Retrieval triggers** (see `eval_utils.findValidActionWithSystem2` and `amm/retrieval.py`):

| Trigger | When | Retrieval | Use |
|---------|------|-------------|-----|
| **T1** | Swift top-1 invalid | S1 (success) → Swift retry; S2 (success+near-miss) → Swift retry | `build_swift_memories_block` in Swift prompt |
| **T2** | Stagnation without Swift failure | S1 / S2 at configured `cycles_without_progress` | Off by default (`enable_t2_retrieval=False`) |
| **T3** | Failed / invalid action history | Avoidance (**B**) | Bundled into T4 when `failed_messages` non-empty |
| **T4** | Sage invoked | S2 + optional **B** | `build_sage_planning_memories_block` in planning prompt |

**Critic memory:** `amm/retrieve_for_critic.py` can attach a compact episodic block to SRM Critic prompts when stagnation coincides with failure signals.

**Config:** `amm/config.py` (`DEFAULT_CONFIG`) — feature flags, milestone thresholds, T2 stagnation steps.

---

## Self-Reflection Module (SRM)

| Component | File | Purpose |
|-----------|------|---------|
| **SRM Gate** | `srm/srm_gate.py` | Final pre-execution check for Swift, Sage buffer, and Critic actions |
| **Action validator** | `srm/action_validator.py` | Parse NL/func actions; check arity, focus rules, valid-action membership, no-ops |
| **Action repairer** | `srm/action_repairer.py` | Deterministic repairs (aliases, inventory focus, buffer fallbacks) |
| **Stagnation detector** | `srm/stagnation.py` | `SAME_OBS`, `REPEAT_ACTION`, `REPEAT_VERB`, `MOVE_SPAM` signals |
| **Critic** | `srm/critic.py` | LLM proposes up to 5 verbatim valid actions to break stagnation |

SRM does **not** replace Sage planning; it constrains what gets executed and intervenes when progress stalls.

---

## Evaluation profiles

`run_eval_fast_slow.sh` supports mutually exclusive modes (passed through to `eval_agent_fast_slow.py`):

| Profile | Flags | AMM | SRM |
|---------|-------|-----|-----|
| **baseline-only** | (default) | off | off |
| **baseline+AMM** | `--use-amm` | on | off |
| **baseline+SRM** | `--use-srm` | off | on |
| **full** | `--use-full` | on | on |

Additional ablations:

- `--amm-write-only` — write EMs, no retrieval / prompt injection  
- `--disable-swift-injection` — AMM on, but skip Swift T1 memory blocks (Sage/Critic memory unchanged)

Logs are written under `fast_slow_logs/<split>_<model>_<profile>/` (per-task `.log`, `-score.txt`, episode `.json`).

### Quick single-task debug

```bash
# Terminal 1: start vLLM for Sage / Critic
./start_vllm.sh 8000 0

# Terminal 2: one task / variation
export LETTA_API_TOKEN=...   # required for --use-amm or --use-full
export LETTA_AGENT_ID=...

### Full benchmark sweep

```bash
bash run_eval_fast_slow.sh 1 --use-full          # 1 GPU
bash run_eval_fast_slow.sh 2 --use-amm           # 4 GPUs, AMM-only ablation
bash run_eval_fast_slow.sh 1 --use-full
```

## Installation

### 1. Environment

```bash
conda create -n swiftsage-ext python=3.8 pip
conda activate swiftsage-ext
pip install scienceworld==1.1.3
pip install -r requirements.txt
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c conda-forge openjdk   # if needed for ScienceWorld JVM
```

### 2. Swift checkpoint

Use the public SwiftSage checkpoint (no local training required):

- https://huggingface.co/yuchenlin/swift_sw  

Default: `--lm_path yuchenlin/swift_sw`.

To retrain Swift (optional): see `fast_agent/ds_train.sh` and `data_utils/data_convert.py`.

### 3. Sage / Critic LLM (vLLM)

```bash
./start_vllm.sh 8000 0
export VLLM_BASE_URL=http://localhost:8000
export QWEN_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct   # optional override
```

### 4. AMM (Letta) — only for `--use-amm` / `--use-full`

```bash
export LETTA_API_TOKEN=...
export LETTA_AGENT_ID=...
```

Create a Letta agent configured for episodic memory / passage search as used in `amm/client_letta.py`.

### 5. ScienceWorld JAR

Point `--jar_path` to the ScienceWorld jar if not using the package default (see [ScienceWorld install](https://github.com/allenai/ScienceWorld)).

---

## Key CLI arguments

| Argument | Default | Notes |
|----------|---------|--------|
| `--task_nums` | — | Comma-separated task indices (0–29) |
| `--set` | `test_mini` | `test` (10 vars/task), `dev`, `test_mini`, etc. |
| `--slow_agent` | on | Enable Sage path when Swift heuristics allow |
| `--use_amm` | off | Enable AMM (or use `--use-amm` / `--use-full` via shell) |
| `--disable_srm` | on | SRM off unless `--use-srm` / `--use-full` |
| `--use_memory_planning` | on | Sage planning uses T4 episodic block |
| `--amm_write_only` | off | Writes only, no retrieval |
| `--disable-swift-injection` | off | T1 Swift memory injection off |
| `--sbert` | on | SBERT fallback when matching valid actions |
| `--beams` | 5 | Swift decoding beams |
| `--env_step_limit` | 300 | Per-episode step cap (runner may use `2×` internally) |

---

## Baselines

Scripts under `baselines/` reproduce comparisons from the SwiftSage paper setting where applicable:

- `eval_agent_saycan.py`, `eval_agent_reflexion.py`, `eval_agent_fast_only.py`  
- Shared utilities: `eval_utils.findValidActionNew`, etc.

See also the official [ScienceWorld](https://github.com/allenai/ScienceWorld) baselines.

---

## Acknowledgments

This codebase builds on [SwiftSage](https://github.com/yuchenlin/SwiftSage) (Allen AI / USC INK) and the [ScienceWorld](https://sciworld.apps.allenai.org) benchmark. Episodic memory is stored and retrieved via [Letta](https://github.com/letta-ai/letta). System 2 reasoning uses [Qwen2.5](https://huggingface.co/Qwen) served with [vLLM](https://github.com/vllm-project/vllm).
