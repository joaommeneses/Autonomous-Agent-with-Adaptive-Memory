# TODO (Prioritized Backlog)

## P0 — Must do first
- [ ] Map SwiftSage entrypoints:
  - [ ] Where SWIFT proposes action
  - [ ] Where “action buffer not useful” is decided
  - [ ] Where SAGE is invoked & buffer executed
  - [ ] Where environment executes actions / returns obs, reward
- [ ] Create `amm/` package with `__init__.py`, `config.py`, `schema.py`, `client_letta.py` (or `store_faiss.py`), `writer.py`, `retriever.py`, `rerank.py`, `mmr.py`, `skills.py`, `working_memory.py`, `instrumentation.py`
- [ ] Implement **working memory** struct (in RAM) with fields: goal signature, room, inventory tokens, obs keyphrases, preconds satisfied/missing, last actions, cycles_without_progress, difficulty_score, last_retrieved_ids
- [ ] Implement **schema** & **writer**:
  - [ ] SUCCESS / NEARMISS / AVOIDANCE records
  - [ ] De-dup via SimHash/LSH
  - [ ] TTL for AVOIDANCE
- [ ] Implement **retriever**:
  - [ ] Query builder from working memory
  - [ ] Letta semantic search OR FAISS sidecar (pick path)
  - [ ] Return raw candidate objects

## P1 — Retrieval quality & controller integration
- [ ] Implement hybrid re-ranker (cosine + goal overlap + success prior + recency)
- [ ] Implement MMR diversity selection
- [ ] Implement difficulty scoring & budgeted K′
- [ ] Implement distillation (NL → compact hints)
- [ ] Pre-SWIFT hook: pass hints; SWIFT reproposes
- [ ] Post-exec write pipeline (gated policy)
- [ ] Unit tests for rerank/MMR/distillation

## P2 — Skills
- [ ] Promotion rule (≥3 consistent successes on same subgoal template)
- [ ] Skill schema (name, templates, preconds, steps, effects, failure_modes)
- [ ] Matcher (template similarity + precondition check)
- [ ] Controller: check skills before retrieval; if preconds missing, emit as hints

## P3 — Instrumentation & ablations
- [ ] Logging: success/steps, invalid-action rate, recovery, SAGE calls, tokens, memory/skill hit-rate, write counts & dedup rate
- [ ] Config surfaces (weights, K/K′, promotion threshold, TTL)
- [ ] Benchmark scripts & ablation toggles

## P4 — Docs & cleanup
- [ ] README for `amm/` + architecture diagram
- [ ] Update project docs (this plan, specs, open questions)
- [ ] Prep SRM surface (no implementation)

## Nice-to-haves / future
- [ ] Cross-encoder for goal_overlap (replace token overlap)
- [ ] Local small model for distillation to reduce API calls
- [ ] Optional local FAISS even if Letta supports search (for deterministic tests)
