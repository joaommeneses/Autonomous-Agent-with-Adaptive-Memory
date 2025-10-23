## integration_swiftsage.md
```markdown
# Integration with SwiftSage

> We avoid large refactors. We add small hooks around existing controller steps.

## Hook points (conceptual)
1) **Pre-SWIFT assist**  
   - If controller judges SWIFT’s initial proposal “weak” (same heuristic used today), call:
     - `AMM.retrieve(working_mem, top_k=20) → candidates`
     - `AMM.rerank_and_select(candidates, working_mem) → hints (K′=3/5/7)`
     - `AMM.match_skills(working_mem) → best_skill?`
   - If `best_skill` applicable (preconditions satisfied) → **execute skill** (populate SWIFT with the steps, or treat as macro-action).
   - Else append **distilled hints** to SWIFT prompt → SWIFT re-proposes.

2) **SAGE escalation (unchanged trigger)**
   - Provide SAGE with: goal signature, latest obs, **distilled hints**, and **skill preconditions** (if any missing).
   - Execute SAGE’s proposed next step (or short buffer) normally.

3) **Post-execution write-back**
   - Update `working_memory` (progress/stall).
   - Call `AMM.write(record)` with **gated policy**:
     - `write_success` if reward>0 or subgoal flips
     - `write_nearmiss` if credible progress (cap)
     - `write_avoidance` on hard precondition violation / repeated identical fail
   - De-dup merges and counters bump.
   - If promotion conditions met → `AMM.promote_to_skill(...)`.

## Minimal function surface (suggested)
```python
# amm/working_memory.py
class WorkingMemory:
    def reset(...): ...
    def update_from_env(obs, reward, valid_actions, ...): ...
    def compute_difficulty(...): ...

# amm/retriever.py
def build_query(wm: WorkingMemory) -> str: ...
def retrieve_candidates(query: str, top_k:int=20) -> list[Memory]: ...

# amm/rerank.py
def hybrid_score(query, memory, now_ts) -> float: ...
def mmr_select(candidates, k_prime:int, lambda_div:float=0.4) -> list[Memory]: ...

# amm/distill.py
def memory_to_hint(memory, wm) -> dict:  # {"hint": "...", "precond": "..."} ...

# amm/skills.py
def match_skill(wm) -> Optional[Skill]: ...
def maybe_promote_to_skill(cluster: list[Memory]) -> Optional[Skill]: ...

# amm/writer.py
def write_success(...): ...
def write_nearmiss(...): ...
def write_avoidance(...): ...

ScienceWorld utilities (where available)

valid_actions(state) to check basic validity fast.

Subgoal state to detect flips.

Canonicalization of objects/rooms for stable fingerprints.