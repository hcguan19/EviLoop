# Web Agent Experiment Summary (2026-07-31)

## 1. Evaluation scope

- Seed: 17
- Model for our method and AgentOccam: `qwen3-vl-32b-instruct`
- Context limit: 16,384 tokens
- WebArena-Verified frozen single-site tasks in this run:
  - Shopping Admin: 55
  - Reddit: 42
  - GitLab: 57
  - Total: 154 tasks per method
- Total attempted method-task runs: 462
- Official evaluator errors: 0

All runtime failures are counted as failures in the strict end-to-end rate.

## 2. Main valid comparison on 154 tasks

| Method | Shopping Admin | Reddit | GitLab | Overall | Runtime failures |
|---|---:|---:|---:|---:|---:|
| Ours v4.9 | 8/55 (14.5%) | 14/42 (33.3%) | 7/57 (12.3%) | **29/154 (18.8%)** | 3 |
| AgentOccam official core + Qwen adapter | 2/55 (3.6%) | 3/42 (7.1%) | **9/57 (15.8%)** | 14/154 (9.1%) | 61 |
| Browser Use original + corrected Qwen adapter | 2/55 (3.6%) | 2/42 (4.8%) | 3/57 (5.3%) | 7/154 (4.5%) | 0 |

Our method improves the strict overall success rate by 9.7 percentage points and reaches 2.07 times the AgentOccam rate. It is strongest on Shopping Admin and Reddit; AgentOccam remains stronger on GitLab.

On paired tasks, our method wins 23 tasks that AgentOccam misses, AgentOccam wins 8 tasks that ours misses, both solve 6, and neither solves 117. The exact McNemar test on the 31 discordant pairs gives `p=0.010674`.

## 3. Conditional scoring and reliability

| Method | Entered official evaluator | Success among evaluated | Main runtime failure causes |
|---|---:|---:|---|
| Ours v4.9 | 151/154 | 29/151 (19.2%) | 2 exceptions, 1 incomplete output |
| AgentOccam | 93/154 | 14/93 (15.1%) | 41 context overflows, 7 browser-action timeouts, 1 connection error, 12 incomplete outputs |
| Browser Use original | 154/154 | 7/154 (4.5%) | No final runtime failures after retrying one cleanup timeout |

The strict advantage therefore contains two effects: higher task-solving accuracy and much higher execution reliability under the shared 16K context budget. This distinction should be explicit in the paper.

## 4. Efficiency metrics

| Method | Metric coverage | Avg. time | Avg. tokens | Avg. steps |
|---|---:|---:|---:|---:|
| Ours v4.9 | 148/154 | 271.1 s | 272,568 | 22.4 |
| AgentOccam | 93/154 | 349.4 s | 116,394 | 9.5 |
| Browser Use original | 154/154 | 293.2 s | 282,455 | 21.5 |

Our method is more reliable and faster on average among recorded runs, but consumes more tokens and steps. AgentOccam's averages only cover its 93 completed runs, so the efficiency comparison is not survivorship-neutral.

## 5. Browser Use baseline correction audit

The earlier apparent Browser Use result of `0/154` is invalid and must not be used. Its metadata records `qwen-vl-plus`, while the local server only exposed `qwen3-vl-32b-instruct`. Logs show repeated HTTP 404 responses, zero consumed tokens, six failed calls per task, and termination without a structured response.

The corrected rerun completed all 154 frozen tasks with `QWEN_CHAT_MODEL=qwen3-vl-32b-instruct`, seed 17, the same task lists and official evaluator, and a matched 16K context budget. The valid result is `7/154 (4.5%)`. The adapter retains the original Browser Use agent loop and only supplies login/HAR/evaluator integration plus runtime limits (`use_vision=auto`, low-detail vision, 6,000-character clickable DOM, and six history items).

## 6. Existing Shopping subset

The earlier WebArena-Verified Shopping run used 50 frozen tasks:

| Method | Strict result | Notes |
|---|---:|---|
| Ours v4.9 | **17/50 (34.0%)** | All 50 officially evaluated |
| AgentOccam official core + Qwen adapter | 3/50 (6.0%) | Only 40 entered the evaluator; 10 runtime failures |
| Browser Use original + Qwen | 4/50 (8.0%) | Valid Qwen3-VL model configuration in this separate run |

Combining our Shopping 50 with the completed 154-task run gives `46/204 = 22.5%` across the four frozen single-site subsets. This is not the full 233-task four-site set because six additional Shopping tasks and 23 multi-site tasks are not included.

## 7. Existing WebShop and MiniWoB++ fixed-50 results

| Method | WebShop | MiniWoB++ |
|---|---:|---:|
| Ours v4.9 | 32/50 strict (64.0%); 32/48 completed (66.7%) | **48/50 (96.0%)** |
| AgentOccam adapted | 12/50 strict (24.0%); 12/48 completed (25.0%) | 36/50 (72.0%) |

These AgentOccam results use an adapted design in the Browser Use harness, whereas the WebArena table uses AgentOccam's official core prompts, action space, parser, observation pruning, and loop with a Qwen/WebArena adapter. They must be labeled separately.

## 8. Paper-ready conclusions and remaining work

Current evidence supports the claim that the proposed method improves cross-site end-to-end robustness, especially through compact history, evidence-focused execution, and reduced context-overflow failures. It does not yet support a final paper main table because:

1. AgentOccam context-overflow failures should be reported and optionally rerun with a matched larger context as a sensitivity experiment.
2. At least two additional seeds are needed for the main methods, or confidence intervals must be reported with the single-seed limitation made explicit.
3. GitLab failures need targeted, domain-general analysis because AgentOccam currently leads there.
4. The six remaining Shopping tasks and 23 multi-site tasks should be evaluated if claiming coverage of the complete four-site frozen set.
5. Ablations should isolate memory/compaction, evidence routing, verification gate, candidate reranking, and recovery policy.

## 9. Artifact locations

Raw trajectories and environment databases are intentionally excluded from the public
repository. Frozen task manifests belong under `evaluation/tasks/`, and new outputs
should be written under the ignored `benchmark_results/` directory.
