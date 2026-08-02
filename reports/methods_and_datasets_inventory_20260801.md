# Methods and Datasets Inventory (2026-08-01)

## Method inventory

| Method | Local official code | Shared-harness result | Result status |
|---|---|---|---|
| Proposed method (v4.9) | Local implementation | MiniWoB++, WebShop, WebArena-Verified | Main method |
| Original Browser Use | Yes | MiniWoB++, WebShop, WebArena-Verified | Valid local baseline |
| Reflexion | Yes | MiniWoB++, WebShop | Adapted, not an official reproduction |
| ADaPT | Yes | MiniWoB++, WebShop | Adapted; upstream repository is incomplete for all paper experiments |
| AgentOccam | Yes | MiniWoB++, WebShop; WebArena-Verified | Adapted on MiniWoB++/WebShop; official core plus Qwen/evaluator adapter on WebArena |
| AdaPlanner | Yes | MiniWoB++, WebShop | Adapted, not an official reproduction |
| WebOperator | Yes | MiniWoB++, WebShop | Adapted, not an official reproduction |
| WebChallenger | Yes | MiniWoB++, WebShop | Adapted, not an official reproduction |
| WebDART | No verified author repository | MiniWoB++, WebShop | Paper-derived adaptation only |
| Mind2Web / MindAct | Yes | None in the shared live-browser table | Official code downloaded; inference not run |
| WebLINX | Yes | None | Official code downloaded; data/weights not installed |
| SeeAct | Yes | None | Official code downloaded; released-model protocol not run |
| Agent Workflow Memory | Yes | None | Official code downloaded; official offline/online pipeline not run |
| Webwright | Yes | None | Official code downloaded; local benchmark run pending |
| OpAgent | Yes | None | Official code downloaded; trained weights/system reproduction pending |

## Dataset inventory

| Dataset or task collection | Local amount | Environment status | Formal result status |
|---|---:|---|---|
| MiniWoB++ | 130 frozen task types | Local environment available | Formal fixed-50 and older multi-seed results exist |
| WebShop | 1,000 local tasks | Local environment available | Formal fixed-50 and older multi-seed results exist |
| WebArena-Verified Hard | 258 official hard tasks | Four sites deployed remotely | Formal results exist on 204 single-site tasks across separate runs |
| WebArena-Verified four-site set | 233 tasks: 210 single-site + 23 multi-site | Shopping, Shopping Admin, Reddit, GitLab deployed | Shopping 50 plus Admin 55, Reddit 42, GitLab 57 evaluated; six Shopping and 23 multi-site tasks remain |
| Mind2Web | 1,009 converted YAML tasks; 341 in the mixed-2000 set | Offline data available; live sites are unstable | No paper-grade shared-harness result |
| WebArena original | 506 converted tasks in the mixed-2000 set | Repository downloaded; original full environment not deployed | No formal result; do not mix with WebArena-Verified |
| VisualWebArena | 341 converted tasks | Repository downloaded; visual environment not deployed | No formal result |
| WorkArena | 341 converted tasks | Task files only; ServiceNow environment not deployed | No formal result |
| OSWorld | 341 converted tasks | Repository downloaded; desktop VM environment not deployed | No formal result |
| Stable-local mixed set | 1,130 tasks | Local emulation collection | Engineering/debug result only, not a public benchmark |
| Downloaded mixed set | 2,000 tasks | Task files only | Engineering inventory, not a standalone dataset |
| DeepShop | Not present in the verified local inventory | Not deployed | No result |

The mixed-2000 directory contains WebArena 506, OSWorld 341, WorkArena 341, Mind2Web 341, VisualWebArena 341, and MiniWoB++ 130 tasks.

## Archived shared 100-task comparison

This table uses 50 MiniWoB++ and 50 WebShop tasks per method-seed pair. The first three methods have three seeds; the remaining adapted methods have one seed.

| Method | Samples / seeds | Overall | MiniWoB++ | WebShop |
|---|---:|---:|---:|---:|
| Proposed method | 300 / 3 | **69.0%** | 71.3% | **66.7%** |
| Reflexion adapted | 300 / 3 | 51.0% | **78.0%** | 24.0% |
| Original Browser Use | 300 / 3 | 44.7% | 67.3% | 22.0% |
| ADaPT adapted | 100 / 1 | 47.0% | 76.0% | 18.0% |
| AgentOccam adapted | 100 / 1 | 46.0% | 70.0% | 22.0% |
| WebOperator adapted | 100 / 1 | 45.0% | 72.0% | 18.0% |
| WebDART adapted | 100 / 1 | 40.0% | 62.0% | 18.0% |
| WebChallenger adapted | 100 / 1 | 39.0% | 66.0% | 12.0% |
| AdaPlanner adapted | 100 / 1 | 34.0% | 42.0% | 26.0% |

## WebArena-Verified same-task comparison

These results use the same 154 frozen Shopping Admin, Reddit, and GitLab tasks with seed 17 and Qwen3-VL-32B-Instruct.

| Method | Shopping Admin | Reddit | GitLab | Overall |
|---|---:|---:|---:|---:|
| Proposed method v4.9 | **8/55 (14.5%)** | **14/42 (33.3%)** | 7/57 (12.3%) | **29/154 (18.8%)** |
| AgentOccam official core + Qwen adapter | 2/55 (3.6%) | 3/42 (7.1%) | **9/57 (15.8%)** | 14/154 (9.1%) |
| Original Browser Use + corrected Qwen adapter | 2/55 (3.6%) | 2/42 (4.8%) | 3/57 (5.3%) | 7/154 (4.5%) |

## WebArena-Verified Shopping-50

| Method | Result |
|---|---:|
| Proposed method v4.9 | **17/50 (34.0%)** |
| Original Browser Use | 4/50 (8.0%) |
| AgentOccam official core + Qwen adapter | 3/50 (6.0%) |

## Reporting rules

1. Keep official-system results separate from shared-Qwen adapted results.
2. Never label WebDART adapted as an official reproduction.
3. Do not combine WebArena original and WebArena-Verified scores.
4. Treat the mixed-2000 and stable-local-1130 directories as engineering collections, not new public datasets.
5. Mark the latest single-seed development results separately from the archived three-seed table.
