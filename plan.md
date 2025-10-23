# AMM + SwiftSage: Implementation Plan

## 0) One-paragraph summary
We extend **SwiftSage** (SWIFT fast policy + SAGE slow planner) with an **Adaptive Memory Module (AMM)** that stores and retrieves high-utility episodic knowledge and distilled “skills” to assist SWIFT before escalating to SAGE. The AMM writes only informative memories (success, rare near-miss, avoidance) and retrieves **diverse, budgeted hints** matched to the current goal/state. Everything is **inference-time** and **controller-level** (no model training, no logits). SRM (Self-Reflection Module) is **out of scope for the first milestone** and will be layered later.

## 1) Baseline system (SwiftSage)
- **SWIFT**: T5/Flan-T5 fast policy that proposes next action from recent context.
- **SAGE**: Gemini-2.5-flash (or similar) planner used when controller deems the step “hard” or the action buffer is “not useful”.
- **Controller**: Orchestrates SWIFT→SAGE switching, holds an action buffer, executes actions in **ScienceWorld** environment.

> We DO NOT modify SWIFT/SAGE internals. We add a memory layer and minimal controller hooks.

## 2) Adaptive Memory Module (AMM) — concept
- **Memory types (tags)**:
  - `episodic_success`: steps that clearly advanced a subgoal.
  - `episodic_nearmiss`: rare “almost there” states (reward=0 but credible progress).
  - `avoidance`: known bad/invalid patterns (precondition violations, repeated failures, invalid action-state).
  - `skills`: distilled procedures promoted from repeated successes (macro-actions).
- **Working memory (scratchpad)**: transient per-episode view (goal signature, room, last obs/inventory keyphrases, preconditions satisfied/missing, last actions, cycles without progress, difficulty score).

**Write policy** (noise-controlled):
- SUCCESS on reward>0 OR objective subgoal flips; **gated** to material progress.
- NEARMISS sparingly (cap = 1 per (subgoal template, room) window).
- AVOIDANCE only for hard precondition violations or repeated same failure; TTL expiry.
- De-duplicate near-identical entries (SimHash/LSH); bump counters for reused memories.

**Retrieval** (assist SWIFT before SAGE):
- Query string = `[GOAL]+[ROOM]+[INVENTORY]+[OBS KEYPHRASES]+[PRECONDS SAT/MISS]+[LAST FAIL?]`.
- Fetch top-K from `episodic_success` (+ a few `nearmiss`), re-rank with hybrid score:
  - relevance (cosine) + goal overlap + success prior + mild recency.
- Diversity (MMR) then **budgeted K′** hints based on difficulty (3/5/7).
- Distill each selected memory to one **actionable hint** (+ optional precondition).
- Prefer **skills** when preconditions are satisfied (execute skill); else pass hints to SWIFT.

## 3) Control flow (high level)
1. SWIFT proposes → if weak: AMM retrieval → distilled hints → SWIFT re-propose.
2. If still weak: escalate to SAGE with goal + obs + distilled hints (+ matching skill).
3. Execute action → update working memory → gated write to AMM → de-dup → (optional) skill promotion.

## 4) Constraints & principles
- **Inference-time only**, minimal invasive changes, feature-flagged rollout.
- ScienceWorld-aware (valid-action lists, subgoals, canonical object names).
- Config-driven thresholds (weights, K/K′, TTL, skill promotion count).
- Strong logging & ablations; correctness > cleverness; efficiency tracked via Accuracy-per-Compute.
- Extending SwiftSage, so existing structures, data etc... will be considered when implementing.

## 5) Deliverables (phase 1)
- AMM library (`amm/`) with: schema, Letta connection (already implemented so we will refactor it for the amm/ folder instead of main code file), writer, retriever, reranker (MMR), skills, working memory, config, instrumentation.
- Controller hooks for: pre-SWIFT assist; post-exec writes; optional skill invocation.
- Metrics: success/steps, invalid-action rate, SAGE calls, tokens, memory/skill hit rates.

## 6) What is explicitly out-of-scope now
- SRM micro-deliberation, verifier, or stall detector (will follow AMM).
- Any RL/SFT training (possible future work).
