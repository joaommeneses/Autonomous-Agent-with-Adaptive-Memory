# Baseline Swift Restoration - Implementation Summary

## Overview
Restored baseline Swift behavior with feature flags to enable/disable AMM and Sage architecture additively.

## Feature Flags Added

### `--use_amm` (default: True)
- When `False`: Disables all AMM operations (retrieval, writing, memory injection)
- When `True`: Enables AMM retrieval and writing (requires LETTA_API_TOKEN and LETTA_AGENT_ID)

### `--use_sage` (default: True)  
- When `False`: Uses baseline Swift-only path (`findValidActionNew`)
- When `True`: Uses architecture path (`findValidActionWithSystem2` with Sage/System 2)
- Requires `slow_agent=True` to be effective

## Baseline Mode Behavior (`use_amm=False`, `use_sage=False`)

### Swift Execution Path
1. **Prompt Construction**: Uses `compose_instance_v4()` - no memory blocks injected
2. **Model Call**: `get_model_output()` with baseline params (beams=5, max_length=16)
3. **Action Selection**: `findValidActionNew()` - exact match → SBERT → Jaccard fallback
4. **No AMM Calls**: No retrieval, no writing, no memory injection
5. **No Sage Calls**: No System 2 planning/grounding

### Key Changes
- AMM initialization gated behind `use_amm` flag
- AMM writing gated behind `use_amm` flag  
- Sage/System2 path gated behind `use_sage` flag
- Baseline Swift path uses `findValidActionNew` (no AMM, no Sage)
- Architecture path uses `findValidActionWithSystem2` (with AMM when enabled)

## Architecture Mode Behavior (`use_amm=True`, `use_sage=True`)

### Additive Features
- **Memory Retrieval**: T1/T2/T3/T4 triggers active (when `amm_client` is not None)
- **Memory Writing**: Post-step memory writing active
- **Sage Integration**: System 2 planning/grounding active
- **Prompt Augmentation**: Memory blocks injected before baseline markers (additive, doesn't change baseline structure)

## Reset Locations for `swift_failure_count`

1. **After buffer action execution** (Line 415): Any action from buffer breaks streak
2. **When System 2 is used** (Line 591): System 2 takeover breaks streak (only when `use_sage=True`)
3. **When force_system_2=True** (Line 512): Swift not expected to operate (only when `use_sage=True`)
4. **When found_valid_in_top=True** (Line 508): Swift found valid action (only when `use_sage=True`)

## Testing Baseline Mode

To run in baseline Swift-only mode:
```bash
python eval_agent_fast_slow.py \
  --use_amm=False \
  --use_sage=False \
  --slow_agent=False \
  ...
```

This ensures:
- No AMM initialization
- No AMM retrieval/writing
- No Sage/System2 calls
- Pure baseline Swift behavior with `findValidActionNew`

## Files Modified

- `eval_agent_fast_slow.py`: Added feature flags, gated AMM/Sage operations, added baseline Swift path

## Files NOT Modified (as requested)

- `amm/*`: No changes to AMM modules
- `llm/*`: No changes to LLM modules  
- `eval_utils.py`: No changes (already handles `amm_client=None` correctly)


