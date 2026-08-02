# WebArena Fixed-6 Regression: v4.5 vs v4.6

## Protocol

- Dataset: WebArena-Verified
- Site: Shopping
- Fixed task IDs: 50, 269, 337, 466, 507, 571
- Seed: 17
- Official evaluator: WebArena-Verified 1.2.3
- Database isolation: clean mutable-state snapshot restored before every task
- Model: qwen3-vl-32b-instruct

## Results

| Version | Passed | Accuracy | Avg. steps | Avg. tokens | Avg. runtime |
|---|---:|---:|---:|---:|---:|
| v4.5 | 3 / 6 | 50.0% | 18.8 | 224,447 | 207.7 s |
| v4.6 | 4 / 6 | 66.7% | 21.2 | 271,552 | 233.9 s |

| Task | Capability | v4.5 | v4.6 | v4.6 steps | v4.6 tokens |
|---:|---|---:|---:|---:|---:|
| 50 | Exhaustive order aggregation | Pass | Pass | 6 | 69,666 |
| 269 | Hierarchical filtered navigation | Fail | Fail | 25 | 324,144 |
| 337 | Long-history entity retrieval | Fail | Pass | 29 | 363,454 |
| 466 | Exact-item wish-list mutation | Pass | Pass | 5 | 56,836 |
| 507 | Constrained product purchase | Fail | Fail | 51 | 693,361 |
| 571 | Multi-field address mutation | Pass | Pass | 11 | 121,848 |

## Validated Improvements

1. Redirect-safe HAR collection preserves the original request in
   `POST -> 302 -> GET` chains. This converted tasks 466 and 571 from
   evaluator false negatives into official passes.
2. Strong visible persisted-state evidence prevents successful mutation
   tasks from looping when the live HAR observer has not finalized a request.
3. A larger structured-evidence ledger and deduplicated detail expansion
   preserve early detail pages during long retrieval. This converted task 337
   from `null` to the correct date.
4. Numbers embedded in exact entity names are no longer automatically treated
   as quantities.

## Remaining Failures

### Task 269

The agent reached a semantically equivalent faceted listing with the exact
`price=0-25` constraint, but the official evaluator required the canonical
hierarchical category URL. The completion recovery ranked product links above
the hidden-until-hover category menu link and repeatedly left the valid
listing.

Next fix: enforce category-shaped link ranking and make hierarchical recovery
hover visible top-level menus before selecting a newly visible child link.

### Task 507

The v4.6 trajectory entered a broad search result page and repeated upward
scrolling more than twenty times while looking for a price filter. Increasing
the budget therefore increased cost without increasing useful work.

Next fix: turn loop detection into an execution constraint. After two
state-equivalent scrolls, block further identical scroll actions and require a
different tool or navigation strategy. Long purchase tasks should receive
extra steps only while evidence or satisfied constraints are increasing.

## Conclusion

v4.6 reaches the requested fixed-set threshold of 4/6 (66.7%) while retaining
all three v4.5 passes. The gain is attributable to a general evidence-retention
change rather than a task-specific answer rule. The next iteration should
target action-level stagnation and hierarchical menu grounding; simply raising
the step budget is not justified by the v4.6 cost profile.
