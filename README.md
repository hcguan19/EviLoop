# EviLoop

**Evidence-Grounded Closed-Loop Web Agent**

EviLoop is a general web-agent method built on top of
[Browser Use](https://github.com/browser-use/browser-use). It keeps the agent loop,
but makes each loop evidence-aware: compact HTML/DOM observations are grounded to
visible controls, candidate actions are reranked against task constraints, progress
is stored in structured memory, and irreversible actions are verified before commit.

![EviLoop architecture](reports/figures/general_web_agent_framework_cn.png)

## Method

EviLoop adds five reusable components without reading private benchmark databases or
hard-coding task answers:

1. **HTML-first evidence extraction** prunes noisy DOM content while preserving visible,
   task-relevant controls. Screenshots are used only when visual evidence is necessary.
2. **Semantic candidate reranking** scores observed controls and records against the
   constraints in the user task.
3. **Structured execution memory** tracks completed constraints, failed actions,
   selected candidates, and remaining subgoals across a long trajectory.
4. **Verified commit guard** blocks submit, purchase, or other irreversible actions
   until required visible options and task constraints have been checked.
5. **Budgeted recovery loop** detects stalls, limits repeated actions, and chooses a
   bounded recovery or early termination path.

The implementation is primarily under:

- `browser_use/general_policy/`
- `browser_use/tools/control_state.py`
- `browser_use/tools/generic_actions.py`
- `browser_use/tools/generic_task_runtime.py`
- `browser_use/tools/task_policy.py`
- `browser_use/agent/prompts.py`
- `tests/ci/evaluate_tasks.py`

## Local Results

All rows below use the same frozen tasks within each dataset. These are local results
with Qwen3-VL-32B-Instruct, not claims of official baseline reproduction.

| Dataset | Tasks | EviLoop | AgentOccam adapted | Browser Use |
|---|---:|---:|---:|---:|
| MiniWoB++ | 50 | **48/50 (96.0%)** | 36/50 (72.0%) | 33/50 (66.0%) |
| WebShop | 50 | **32/50 (64.0%)** | 12/50 (24.0%) | 12/50 (24.0%) |
| WebArena Shopping | 50 | **17/50 (34.0%)** | 3/50 (6.0%) | 4/50 (8.0%) |
| WebArena Shopping Admin | 55 | **8/55 (14.5%)** | 2/55 (3.6%) | 2/55 (3.6%) |
| WebArena Reddit | 42 | **14/42 (33.3%)** | 3/42 (7.1%) | 2/42 (4.8%) |
| WebArena GitLab | 57 | 7/57 (12.3%) | **9/57 (15.8%)** | 3/57 (5.3%) |

See [`reports/main_and_ablation_experiments_20260802.md`](reports/main_and_ablation_experiments_20260802.md)
for the experimental scope, caveats, and ablation diagnostics.

## Installation

Python 3.11+ is required.

```powershell
conda create -n eviloop python=3.11 -y
conda activate eviloop
pip install -e .
playwright install chromium
```

Copy `.env.example` to `.env` and provide an OpenAI-compatible chat endpoint. Never
commit the resulting `.env` file.

```dotenv
QWEN_CHAT_BASE_URL=http://127.0.0.1:8001/v1
QWEN_CHAT_API_KEY=replace-me
QWEN_CHAT_MODEL=qwen3-vl-32b-instruct
```

## Fixed-50 Evaluation

The frozen Seed-17 task manifests are included in `evaluation/tasks/`. Start the
corresponding MiniWoB++ or WebShop website first, then run:

```powershell
./scripts/run_eviloop_fixed50.ps1 -Benchmark miniwob
./scripts/run_eviloop_fixed50.ps1 -Benchmark webshop
```

The runner accepts `-Python`, `-BaseUrl`, `-Model`, `-ApiKey`, `-MaxParallel`, and
`-ResultDirectory`. Credentials may also be supplied through the `QWEN_CHAT_*`
environment variables.

WebArena-Verified launchers live under `scripts/webarena_verified/`. WebArena sites
and benchmark databases are intentionally not bundled in this repository.

## Reproducibility Notes

- The included fixed task files are manifests, not benchmark website databases.
- Generated trajectories, screenshots, JSONL outputs, caches, model weights, and API
  credentials are excluded from version control.
- Adapted baselines share the local Browser Use evaluation harness. They are labelled
  as adapted and must not be reported as official system reproductions.
- The current ablation report contains infrastructure-affected diagnostic runs; rerun
  those variants under one stable model service before using them as final paper results.

## Attribution

EviLoop is derived from Browser Use and retains its MIT license. The original Browser
Use README is preserved at `docs/UPSTREAM_BROWSER_USE_README.md`.

## License

[MIT](LICENSE)
