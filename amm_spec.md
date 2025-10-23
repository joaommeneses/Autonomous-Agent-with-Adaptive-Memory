# AMM Specification

## 1) Memory types (tags)
- `episodic_success`: step advanced subgoal (reward>0 OR subgoal flips true).
- `episodic_nearmiss`: rare, reward=0 but credible progress (e.g., precondition became satisfied).
- `avoidance`: invalid/unsafe patterns (precondition violations, repeated same failure). TTL expiry or flip on contradiction.
- `skills`: distilled macro-procedures promoted from repeated successes.

## 2) Record schema (per episodic entry - INITIAL IDEA, CAN AND WILL BE ANALYSED BASED ON SCIENCEWORLD SPECS AND EXISTING STRUCTURES IN OUR CURRENT CODE)
```json
{
  "id": "mem_xxx",
  "type": "success | nearmiss | avoidance | skill",
  "goal_signature": "heat solution to 90C",
  "state_fingerprint": ["room:chem_lab_A","burner:off","stand:available","beaker:present"],
  "preconds_satisfied": ["burner_present","beaker_present"],
  "preconds_missing": ["burner_on"],
  "action_seq": ["take beaker","place beaker on stand"],
  "obs_seq": ["burner is off","stand is in front of you"],
  "inventory_delta": ["+beaker"],
  "summary": "Light burner then place beaker on stand to heat safely.",
  "success_weight": 1,
  "avoid_tags": [],
  "embedding": "…",
  "meta": {"created_ts": 0, "last_seen_ts": 0, "episode_id": "…", "simhash": "…"}
}

3) Write policy

SUCCESS if reward>0 OR objective subgoal flips (or equivalent explicit success signal). Weight increments if later reused or leads to terminal success.

NEARMISS only if credible progress (precondition flip, helpful object placement) and cap to 1 per (subgoal template, room).

AVOIDANCE only for hard precondition violations or repeated identical failure within window; add TTL (Time to Live).

De-dup: compute SimHash/LSH on (goal_signature + summary); merge near-duplicates (increment counts and update last_seen).

4) Working memory (in RAM; not persisted)
{
  "pending_subgoal": "heat solution to 90C",
  "room": "chem_lab_A",
  "inventory": ["beaker","thermometer","ethanol"],
  "last_obs_keyphrases": ["burner off","stand available"],
  "preconds_satisfied": ["burner_present","beaker_present"],
  "preconds_missing": ["burner_on"],
  "last_actions": ["take beaker","place beaker on stand"],
  "last_validator_msg": "",
  "cycles_without_progress": 0,
  "difficulty_score": 0.0,
  "last_progress_ts": 0,
  "last_retrieved_ids": []
}

5) Retrieval
5.1 Query builder
[GOAL] heat solution to 90C
[ROOM] chem_lab_A
[INVENTORY] beaker; thermometer; ethanol
[OBS] burner off; stand available
[PRECONDS_SAT] burner_present; beaker_present
[PRECONDS_MISS] burner_on
[LAST_FAIL] (optional)

5.2 Candidate fetch

Prefer Letta semantic search (top-K=20).

5.3 Hybrid re-rank (score)
score
(
𝑚
)
=
1.0
⋅
cos
(
𝑞
,
𝑚
)
+
0.5
⋅
goal_overlap
+
0.3
⋅
log
⁡
(
1
+
success_weight
)
+
0.2
⋅
recency
score(m)=1.0⋅cos(q,m)+0.5⋅goal_overlap+0.3⋅log(1+success_weight)+0.2⋅recency

goal_overlap: token overlap between goal strings (upgradeable to cross-encoder).

recency: exp(-Δt/τ); τ ≈ 72h.

5.4 Diversity (MMR)

Greedy select with:
λ * score(m) – (1–λ) * max_sim_to_selected(m), λ∈[0.3,0.5].

5.5 Budgeted K′ (by difficulty)

easy (≤0.3) → K′=3

medium (0.3–0.7) → K′=5

hard (>0.7) → K′=6–8 (+ one “avoidance” reminder if relevant)

5.6 Distillation to hints

Prompt template (LLM or lightweight rule):

Given GOAL and OBS, extract one actionable next step and (optionally) one precondition implied by this memory. Output JSON.

6) Skills
6.1 Promotion

If same subgoal template solved ≥3 times → synthesize skill:

{
  "name": "heat_solution",
  "goal_templates": ["heat * to <temp>", "raise temperature of *"],
  "preconds": ["burner_present","beaker_present","burner_on"],
  "steps": ["light burner","place beaker on stand","wait until >= 90C"],
  "effects": ["temperature>=90C"],
  "failure_modes": ["spill if no stand"]
}

6.2 Matching

Template similarity (token overlap or embedding cosine > τ).

Preconditions satisfied in current state; if missing, expose as hints.

7) Config defaults

K=20; K′ = 3/5/7

MMR λ = 0.4

Weights: cos=1.0, goal_overlap=0.5, success_prior=0.3, recency=0.2

NEARMISS cap: 1 per (subgoal template, room)

AVOIDANCE TTL: 50 episodes

Skill promotion: 3 successes