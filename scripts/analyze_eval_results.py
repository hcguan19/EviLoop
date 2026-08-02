"""Analyze Browser Use evaluation JSONL results.

The report separates model/browser failures from website blockers so the
accuracy on the actually evaluable subset is visible.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


Outcome = Literal['success', 'reachable_failed', 'blocked_or_unreachable']


class EvalResult(BaseModel):
	"""One line from tests/ci/evaluate_tasks.py JSONL output."""

	file: str
	success: bool
	explanation: str = ''


class ClassifiedResult(EvalResult):
	"""Evaluation result with an analysis label."""

	outcome: Outcome
	blocker_type: str | None = None
	source: str = 'unknown'
	target: str = 'unknown'


class EvalSummary(BaseModel):
	"""Aggregate evaluation metrics."""

	total: int
	success: int
	reachable_failed: int
	blocked_or_unreachable: int
	evaluable_total: int
	raw_accuracy: float
	evaluable_accuracy: float | None
	blocked_rate: float
	by_blocker_type: dict[str, int] = Field(default_factory=dict)
	by_source: dict[str, dict[str, int]] = Field(default_factory=dict)
	top_blocked_targets: dict[str, int] = Field(default_factory=dict)
	top_failed_targets: dict[str, int] = Field(default_factory=dict)


BLOCKER_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
	('anti_bot', re.compile(r'captcha|cloudflare|security verification|bot detection|human verification', re.I)),
	('access_denied', re.compile(r'access denied|forbidden|403|not authorized|unauthorized', re.I)),
	('connection_error', re.compile(r'err_connection_closed|err_connection_reset|err_name_not_resolved|dns|connection error', re.I)),
	('site_unavailable', re.compile(r'503|service temporarily unavailable|temporarily unavailable|site unavailable|website unavailable|unavailable', re.I)),
	('timeout', re.compile(r'timeout|timed out|took too long', re.I)),
	('benchmark_environment_missing', re.compile(r'__shopping|__classifieds|webarena|workarena|servicenow environment|environment is unavailable', re.I)),
]


def _source_from_file(file_name: str) -> str:
	parts = Path(file_name).stem.split('_', 2)
	if len(parts) < 2:
		return 'unknown'
	return parts[1]


def _target_from_file(file_name: str) -> str:
	stem = Path(file_name).stem
	parts = stem.split('_')
	if len(parts) < 3:
		return 'unknown'
	source = parts[1]
	if source == 'mind2web':
		return parts[2]
	if source == 'miniwob':
		return '_'.join(parts[2:])
	return source


def _classify(result: EvalResult) -> ClassifiedResult:
	source = _source_from_file(result.file)
	target = _target_from_file(result.file)
	if result.success:
		return ClassifiedResult(**result.model_dump(), outcome='success', source=source, target=target)

	for blocker_type, pattern in BLOCKER_PATTERNS:
		if pattern.search(result.explanation):
			return ClassifiedResult(
				**result.model_dump(),
				outcome='blocked_or_unreachable',
				blocker_type=blocker_type,
				source=source,
				target=target,
			)

	return ClassifiedResult(**result.model_dump(), outcome='reachable_failed', source=source, target=target)


def load_results(path: Path) -> list[ClassifiedResult]:
	results: list[ClassifiedResult] = []
	with path.open(encoding='utf-8') as file:
		for line_number, line in enumerate(file, start=1):
			stripped = line.strip()
			if not stripped:
				continue
			try:
				result = EvalResult.model_validate_json(stripped)
			except Exception as error:
				raise ValueError(f'Invalid JSONL result on line {line_number}: {error}') from error
			results.append(_classify(result))
	return results


def summarize(results: list[ClassifiedResult]) -> EvalSummary:
	total = len(results)
	success = sum(1 for result in results if result.outcome == 'success')
	reachable_failed = sum(1 for result in results if result.outcome == 'reachable_failed')
	blocked = sum(1 for result in results if result.outcome == 'blocked_or_unreachable')
	evaluable_total = success + reachable_failed

	by_blocker_type = Counter(
		result.blocker_type or 'unknown' for result in results if result.outcome == 'blocked_or_unreachable'
	)
	by_source: dict[str, Counter[str]] = {}
	for result in results:
		by_source.setdefault(result.source, Counter())[result.outcome] += 1

	top_blocked_targets = Counter(
		result.target for result in results if result.outcome == 'blocked_or_unreachable'
	).most_common(20)
	top_failed_targets = Counter(result.target for result in results if result.outcome == 'reachable_failed').most_common(20)

	return EvalSummary(
		total=total,
		success=success,
		reachable_failed=reachable_failed,
		blocked_or_unreachable=blocked,
		evaluable_total=evaluable_total,
		raw_accuracy=success / total if total else 0.0,
		evaluable_accuracy=success / evaluable_total if evaluable_total else None,
		blocked_rate=blocked / total if total else 0.0,
		by_blocker_type=dict(sorted(by_blocker_type.items())),
		by_source={source: dict(counter) for source, counter in sorted(by_source.items())},
		top_blocked_targets=dict(top_blocked_targets),
		top_failed_targets=dict(top_failed_targets),
	)


def _format_percent(value: float | None) -> str:
	if value is None:
		return 'n/a'
	return f'{value * 100:.2f}%'


def print_report(summary: EvalSummary) -> None:
	print('Browser Use Evaluation Analysis')
	print('===============================')
	print(f'Total completed: {summary.total}')
	print(f'Success: {summary.success}')
	print(f'Reachable failed: {summary.reachable_failed}')
	print(f'Blocked or unreachable: {summary.blocked_or_unreachable}')
	print(f'Evaluable subset: {summary.evaluable_total}')
	print(f'Raw accuracy: {_format_percent(summary.raw_accuracy)}')
	print(f'Evaluable accuracy: {_format_percent(summary.evaluable_accuracy)}')
	print(f'Blocked rate: {_format_percent(summary.blocked_rate)}')

	print('\nBlocker types')
	for name, count in summary.by_blocker_type.items():
		print(f'- {name}: {count}')

	print('\nBy source')
	for source, counts in summary.by_source.items():
		print(f'- {source}: {counts}')

	if summary.top_blocked_targets:
		print('\nTop blocked targets')
		for target, count in summary.top_blocked_targets.items():
			print(f'- {target}: {count}')

	if summary.top_failed_targets:
		print('\nTop reachable-failed targets')
		for target, count in summary.top_failed_targets.items():
			print(f'- {target}: {count}')


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('results_jsonl', type=Path)
	parser.add_argument('--output-json', type=Path, help='Optional path to save the summary JSON.')
	args = parser.parse_args()

	summary = summarize(load_results(args.results_jsonl))
	print_report(summary)
	if args.output_json:
		args.output_json.parent.mkdir(parents=True, exist_ok=True)
		args.output_json.write_text(summary.model_dump_json(indent=2), encoding='utf-8')


if __name__ == '__main__':
	main()
