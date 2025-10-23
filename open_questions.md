# Open Questions / To Decide

## SwiftSage code integration
- Exact file/function names for:
  - SWIFT proposal hook
  - “Action buffer not useful” predicate
  - SAGE invocation sites & buffer execution loop
  - Environment step API (obs/reward retrieval)
- Where to keep `WorkingMemory` instance in controller (class vs module-level).

## Letta/MemGPT API details
- Does Letta expose **semantic search** with `top_k` and filter-by-tag?
  - If **yes**: use it directly, store Letta `id` and fetch bodies by IDs.
  - If **no**: implement FAISS sidecar + Letta by-ID read/write.
- How to store tags/TTL for `avoidance` in Letta (native fields vs metadata).

## Observation & preconditions
- Canonicalization: mapping ScienceWorld object names → normalized tokens.
- Precondition detection: do we have lightweight rules (e.g., “burner_on”) we can infer from obs strings, or do we add a tiny local parser?
- Valid-action acquisition: is there a fast API in our wrapper?

## Distillation
- Which LLM for hint distillation? (SAGE vs small local model)
- Max hints per step (2–3) and final format.

## Skills
- Where to store step sequences: as natural language or compact DSL?
- Execution: push as macro-actions to SWIFT, or call SAGE to execute step-by-step?
- Promotion threshold (start with 3 successes, tune later).

## Config & logging
- Where to centralize thresholds (weights, K/K′, λ, TTL, promotion N)?
- Logging destination (JSONL per episode?); minimal dashboard?

## Performance & limits
- Rate limits for SAGE; batching opportunities for distillation?
- Vector store size expectations; compaction schedule.
