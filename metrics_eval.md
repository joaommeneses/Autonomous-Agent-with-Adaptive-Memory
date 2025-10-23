## metrics_eval.md
```markdown
# Metrics & Evaluation

## Core metrics
- **Success rate** (% tasks completed)
- **Steps-to-success** (lower is better)
- **Invalid-action rate** (% invalid or blocked actions)
- **Recovery rate** (success after avoidable error)

## Efficiency
- **SAGE calls per episode**
- **Tokens per episode** (SAGE + SWIFT prompts)
- **Accuracy-per-Compute** (successes per 1k tokens)

## AMM effectiveness
- **Memory hit-rate** (# times a retrieved hint directly led to a correct next action / # retrievals)
- **Skill hit-rate** (# times a skill executed successfully / # skill checks)
- **Writes by type** (success / nearmiss / avoidance) and **dedup rate**

## Ablations
1) Baseline SwiftSage
2) + AMM write policy (SUCCESS/NEARMISS/AVOIDANCE + de-dup)
3) + Retrieval (hybrid re-rank + MMR + budgeted K′) + hints
4) + Skills (promotion + matcher)

## Reporting
- CSV logs per episode; simple notebooks to plot trends.
- Config snapshot attached to each run (weights, K/K′, thresholds).