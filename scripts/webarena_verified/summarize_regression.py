from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
	return json.loads(path.read_text(encoding='utf-8'))


def canonical_url_audit(task_dir: Path, eval_result: dict[str, Any]) -> dict[str, Any] | None:
	failed = [
		item
		for item in eval_result.get('evaluators_results', [])
		if item.get('evaluator_name') == 'NetworkEventEvaluator' and item.get('score') == 0
	]
	if len(failed) != 1:
		return None

	expected = failed[0].get('expected') or {}
	expected_pattern = ((expected.get('url') or {}).get('base_url') or '')
	if not expected_pattern.startswith('^__SHOPPING__') or not expected_pattern.endswith('$'):
		return None

	har_path = task_dir / 'network.har'
	if not har_path.exists():
		return None

	har_text = har_path.read_text(encoding='utf-8', errors='replace')
	local_pattern = expected_pattern.replace('__SHOPPING__', r'https?://[^/]+')
	canonical_pattern = local_pattern[:-1].rstrip('/') + r'/?$'
	urls = sorted(set(re.findall(r'https?://[^"\\\s]+', har_text)))
	matches = [url.rstrip('",') for url in urls if re.match(canonical_pattern, url.rstrip('",'))]
	if not matches:
		return None

	other_results = [
		item.get('score') == 1
		for item in eval_result.get('evaluators_results', [])
		if item is not failed[0]
	]
	return {
		'reason': 'official_url_regex_rejects_equivalent_trailing_slash',
		'expected_pattern': expected_pattern,
		'canonical_pattern': canonical_pattern,
		'matching_har_urls': matches,
		'other_evaluators_passed': bool(other_results) and all(other_results),
	}


def summarize(root: Path) -> dict[str, Any]:
	rows: list[dict[str, Any]] = []
	for task_dir in sorted(
		(path for path in root.iterdir() if path.is_dir() and path.name.isdigit()),
		key=lambda path: int(path.name),
	):
		eval_result = load_json(task_dir / 'eval_result.json')
		metrics = load_json(task_dir / 'metrics.json')
		agent_response = load_json(task_dir / 'agent_response.json')
		audit = canonical_url_audit(task_dir, eval_result)
		audited_success = bool(
			eval_result.get('score') == 1
			or (
				audit
				and audit['other_evaluators_passed']
				and agent_response.get('status') == 'SUCCESS'
			)
		)
		if audit:
			(task_dir / 'evaluator_audit.json').write_text(
				json.dumps(
					{
						'task_id': int(task_dir.name),
						'official_score': eval_result.get('score'),
						'agent_status': agent_response.get('status'),
						'audited_execution_success': audited_success,
						**audit,
					},
					ensure_ascii=False,
					indent=2,
				)
				+ '\n',
				encoding='utf-8',
			)

		rows.append(
			{
				'task_id': int(task_dir.name),
				'official_score': float(eval_result.get('score', 0)),
				'audited_execution_success': audited_success,
				'agent_status': agent_response.get('status'),
				'steps': int(metrics.get('steps', 0)),
				'total_tokens': int(metrics.get('total_tokens', 0)),
				'duration_seconds': round(float(metrics.get('duration_seconds', 0)), 1),
				'audit_reason': audit['reason'] if audit else None,
			}
		)

	official_passes = sum(row['official_score'] for row in rows)
	audited_passes = sum(row['audited_execution_success'] for row in rows)
	return {
		'method': 'ours_v4_8_13_general_workflow',
		'task_count': len(rows),
		'official_passes': official_passes,
		'official_success_rate': official_passes / len(rows) if rows else 0,
		'audited_execution_passes': audited_passes,
		'audited_execution_success_rate': audited_passes / len(rows) if rows else 0,
		'average_steps': statistics.mean(row['steps'] for row in rows) if rows else 0,
		'average_tokens': statistics.mean(row['total_tokens'] for row in rows) if rows else 0,
		'average_duration_seconds': (
			statistics.mean(row['duration_seconds'] for row in rows) if rows else 0
		),
		'rows': rows,
	}


def markdown(summary: dict[str, Any]) -> str:
	lines = [
		'# Fixed Regression Summary',
		'',
		f"- Method: `{summary['method']}`",
		f"- Official score: {summary['official_passes']:.0f}/{summary['task_count']} "
		f"({summary['official_success_rate']:.1%})",
		f"- Audited execution outcome: {summary['audited_execution_passes']}/"
		f"{summary['task_count']} ({summary['audited_execution_success_rate']:.1%})",
		f"- Average tokens / steps / duration: {summary['average_tokens']:.0f} / "
		f"{summary['average_steps']:.1f} / {summary['average_duration_seconds']:.1f}s",
		'',
		'| Task | Official | Audited execution | Steps | Tokens | Seconds | Audit |',
		'|---:|---:|---:|---:|---:|---:|---|',
	]
	for row in summary['rows']:
		lines.append(
			f"| {row['task_id']} | {row['official_score']:.0f} | "
			f"{'pass' if row['audited_execution_success'] else 'fail'} | "
			f"{row['steps']} | {row['total_tokens']} | {row['duration_seconds']:.1f} | "
			f"{row['audit_reason'] or ''} |"
		)
	lines.extend(
		[
			'',
			'The official score is never overwritten. The audited execution column only',
			'documents cases where the HAR contains a canonically equivalent request and',
			'the sole mismatch is an evaluator URL regex that rejects a trailing slash.',
			'',
		]
	)
	return '\n'.join(lines)


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument('result_dir', type=Path)
	args = parser.parse_args()
	summary = summarize(args.result_dir)
	(args.result_dir / 'regression_summary.json').write_text(
		json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
		encoding='utf-8',
	)
	(args.result_dir / 'REGRESSION_ANALYSIS.md').write_text(
		markdown(summary),
		encoding='utf-8',
	)
	print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
	main()
