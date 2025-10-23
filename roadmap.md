# Roadmap

## Phase 0 — Repo familiarization & scaffold
- Read SwiftSage repo; identify controller loop, action buffer creation, and ScienceWorld env wrapper.
- Add `amm/` package skeleton and config.
- Wire a minimal Letta/MemGPT client (or stub) for reads/writes. I already implemented a simple SUCCESS write on reward > 0 so that can be maintained, but refactored into `amm/` folder.

**Exit criteria**: Can import `amm` and call `AMM.retrieve(...)` (stubbed) and `AMM.write(...)` (stubbed) from the controller without breaking runs.

## Phase 1 — AMM v0 (writes + basic retrieval)
- Implement memory schema & write policy (SUCCESS/NEARMISS/AVOIDANCE + de-dup).
- Implement working memory scratchpad (in RAM).
- Implement retrieval with query via Lett, return raw episodes.

**Exit criteria**: After each action, writes happen correctly; we can fetch K candidates by query and see plausible items.

## Phase 2 — AMM v1 (hybrid re-rank + diversity + hints)
- Hybrid scorer: relevance (cosine), goal overlap, success prior, recency.
- MMR diversity; budgeted K′ by difficulty; distillation to compact hints.
- Controller pre-SWIFT hook: pass hints to SWIFT; unit tests for scorer/MMR.

**Exit criteria**: Measurable increase in SWIFT’s good proposals before SAGE escalation on a small task set.

## Phase 3 — AMM v2 (skills)
- Cluster/promotion: if same subgoal solved ≥3 times, synthesize a `skill` (preconds, steps, effects).
- Skill matcher (goal template + preconditions).
- Controller hook: check skill first; if preconds missing, convert to hints.

**Exit criteria**: Reduced SAGE calls and token use with stable success rate; skill hit-rate > 0 on repeated subgoals.

## Phase 4 — Hardening & eval
- Logging & dashboards; config surfacing; ablations.
- Optimize thresholds; tidy docs; prepare SRM surface for next stage.

**Exit criteria**: Clean ablation report, configs checked in, docs in `docs/` or `.md` files.
