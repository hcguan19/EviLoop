# General Agent Policy And Benchmark Boundaries

## Design rule

The Agent policy never branches on benchmark names. It receives a normalized observation, task constraints, visible candidates, capability routes, and bounded progress memory. Benchmark-specific code is restricted to environment startup, observation conversion, action execution, reset, and official scoring.

## Shared policy

1. Extract task constraints from the user instruction and currently visible controls.
2. Route by observable capabilities: forms, search, candidate lists, options, dynamic controls, files, tables, visual regions, commit controls, and terminal evidence.
3. Rank only candidates exposed in the current observation. Record matched evidence and constraint violations.
4. Detect repeated states, repeated actions, and exhausted candidates; backtrack or replan within fixed bounds.
5. Block irreversible actions while required visible constraints are unresolved.
6. Accept success only from visible terminal evidence or an official environment evaluator.
7. Skip the LLM loop when a uniquely grounded programmatic workflow has already completed the task.

## Dataset adapters

| Dataset | Observation | Execution | Final scoring | Current status |
|---|---|---|---|---|
| MiniWoB++ | DOM | Local Chromium | Official runtime reward | Native and tested |
| WebShop | DOM | Local Chromium | Official environment reward | Native and tested |
| WebArena | DOM | Self-hosted sites | Official evaluator | Environment and scorer pending |
| VisualWebArena | DOM + screenshot | Self-hosted sites | Official evaluator | Environment and scorer pending |
| WorkArena | DOM | Authenticated ServiceNow environment | Official evaluator | Environment, auth, and scorer pending |
| OSWorld | Desktop screenshot/state | Desktop VM | Official state evaluator | Desktop adapter and scorer pending |
| Mind2Web | Recorded DOM/action trajectory | Offline replay | Action/step match metrics | Replay adapter and scorer pending |

Tasks whose official adapter is pending are exploratory only. Their LLM-judge result must not be mixed with official task success in the paper.

## Evidence rules

- No hidden product database, answer key, goal state, or environment reward may enter candidate ranking.
- Programmatic actions must use task text and current visible observations only.
- Every result records the observation modality, execution adapter, scoring adapter, readiness status, policy routes, steps, tokens, duration, and failure category.
- Infrastructure failures and unavailable benchmark environments are rerun or reported separately, never counted as method failures.

## Required experiment slices

- Main results: official evaluator only, fixed task lists, identical model and budget for all replaceable-model baselines.
- Generalization: seen capability combinations versus unseen combinations.
- Ablations: candidate ranking, progress memory, commit guard, grounded completion, visual fallback, and programmatic workflow short-circuit.
- Efficiency: success, official reward, steps, latency, prompt/completion tokens, LLM calls, and programmatic-route hit rate.
- Robustness: at least three seeds for the proposed method and principal baselines; one-seed adapted baselines remain explicitly labeled preliminary.
