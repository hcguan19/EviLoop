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

## 实验结果

除特别说明外，实验使用 Qwen3-VL-32B-Instruct。每张表内部使用相同的冻结任务、
随机种子和评测器。`adapted` 表示在共享 Browser Use harness 中实现论文思想，不能视为
对应方法官方系统的复现结果。

### 主实验一：历史三 Seed 对比

MiniWoB++ 与 WebShop 各包含 50 条任务/Seed。EviLoop、Reflexion 和原版 Browser Use
运行 3 个 Seed，其余 adapted 方法当前运行 1 个 Seed。

| 方法 | 样本 / Seed | 总成功率 | MiniWoB++ | WebShop | 复现类型 |
|---|---:|---:|---:|---:|---|
| **EviLoop** | 300 / 3 | **69.0%** | 71.3% | **66.7%** | 本文方法 |
| Reflexion | 300 / 3 | 51.0% | **78.0%** | 24.0% | Adapted |
| 原版 Browser Use | 300 / 3 | 44.7% | 67.3% | 22.0% | 本地基线 |
| ADaPT | 100 / 1 | 47.0% | 76.0% | 18.0% | Adapted |
| AgentOccam | 100 / 1 | 46.0% | 70.0% | 22.0% | Adapted |
| WebOperator | 100 / 1 | 45.0% | 72.0% | 18.0% | Adapted |
| WebDART | 100 / 1 | 40.0% | 62.0% | 18.0% | Paper-derived adapted |
| WebChallenger | 100 / 1 | 39.0% | 66.0% | 12.0% | Adapted |
| AdaPlanner | 100 / 1 | 34.0% | 42.0% | 26.0% | Adapted |

### 主实验二：最新同任务固定子集（Seed 17）

| 数据集 | 样本数 | **EviLoop** | AgentOccam adapted | 原版 Browser Use |
|---|---:|---:|---:|---:|
| MiniWoB++ | 50 | **48/50（96.0%）** | 36/50（72.0%） | 33/50（66.0%） |
| WebShop | 50 | **32/50（64.0%）** | 12/50（24.0%） | 12/50（24.0%） |
| WebArena Shopping | 50 | **17/50（34.0%）** | 3/50（6.0%） | 4/50（8.0%） |
| WebArena Shopping Admin | 55 | **8/55（14.5%）** | 2/55（3.6%） | 2/55（3.6%） |
| WebArena Reddit | 42 | **14/42（33.3%）** | 3/42（7.1%） | 2/42（4.8%） |
| WebArena GitLab | 57 | 7/57（12.3%） | **9/57（15.8%）** | 3/57（5.3%） |
| **WebArena 四站合计** | **204** | **46/204（22.5%）** | 17/204（8.3%） | 11/204（5.4%） |

在 WebArena Shopping Admin、Reddit 和 GitLab 的 154 条配对任务上，EviLoop 独立
成功 23 条、AgentOccam 独立成功 8 条、共同成功 6 条，McNemar 精确检验
`p=0.010674`。GitLab 是当前例外：AgentOccam 的成功率高于 EviLoop。

历史三 Seed 表与最新固定子集表来自不同代码检查点，因此分别报告，不跨表合并计算。

### WebShop 消融实验（Seed 17）

成功要求官方 WebShop reward `>= 0.99`。有效结账表示 Agent 到达结账结果，但不一定
满足任务的全部商品属性与选项约束；结账精度为成功数/有效结账数。

| 方法变体 | 成功率 | 有效结账 | 结账精度 | 平均 Reward | 平均 Token | 平均步骤 | 平均耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 完整方法 Full | 0/50（0.0%） | 0 | 0.0% | 0.000 | 147,799 | 12.1 | 195.0 秒 |
| 去除 Memory | 0/50（0.0%） | 0 | 0.0% | 0.000 | **19,490** | **7.0** | **46.9 秒** |
| 去除语义重排 | 0/50（0.0%） | 0 | 0.0% | 0.000 | 156,161 | 12.5 | 203.8 秒 |
| 去除购买门禁 | 2/50（4.0%） | 20 | 10.0% | 0.275 | 68,522 | 7.9 | 164.9 秒 |
| 去除验证后自动提交 | 0/50（0.0%） | 0 | 0.0% | 0.000 | 73,366 | 9.7 | 163.4 秒 |
| 单候选恢复 | **17/50（34.0%）** | 18 | **94.4%** | **0.350** | 88,414 | 8.0 | **83.4 秒** |
| 仅通用 Agent | 4/50（8.0%） | **23** | 17.4% | 0.330 | 120,473 | 9.4 | 106.4 秒 |

> **有效性说明：** 本轮 Full、去除 Memory 和去除语义重排等组受到 DOM/Screenshot
> Watchdog 超时、16K 上下文超限及早期 API 中断影响，其中 Full 组有 26 条任务因
> 超时/预算耗尽失败。因此该表是故障诊断结果，不应直接作为论文最终消融结论；正式
> 论文版本需要在稳定基础设施上对相同 50 条任务重新运行受污染组。

完整实验设置、失败分布与论文使用边界见：

- [`reports/main_and_ablation_experiments_20260802.md`](reports/main_and_ablation_experiments_20260802.md)
- [`reports/webshop_ablation_summary_20260802.md`](reports/webshop_ablation_summary_20260802.md)
- [`reports/webarena_experiment_summary_20260731.md`](reports/webarena_experiment_summary_20260731.md)

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
