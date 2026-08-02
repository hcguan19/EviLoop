"""Convert local Mind2Web records into Browser Use agent task YAML files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Mind2WebRecord(BaseModel):
	"""A single processed Mind2Web task record."""

	id: str
	website: str
	domain: str
	subdomain: str | None = None
	confirmed_task: str
	action_reprs: list[str] = Field(default_factory=list)


class AgentTask(BaseModel):
	"""YAML-compatible task consumed by tests/ci/evaluate_tasks.py."""

	name: str
	task: str
	judge_context: list[str]
	max_steps: int


def _slugify(value: str, max_length: int = 80) -> str:
	slug = re.sub(r'[^a-zA-Z0-9]+', '_', value.strip().lower()).strip('_')
	return slug[:max_length] or 'task'


def _yaml_scalar(value: str) -> str:
	escaped = value.replace('\\', '\\\\').replace('"', '\\"')
	return f'"{escaped}"'


def _write_agent_task(path: Path, task: AgentTask) -> None:
	lines = [
		f'name: {_yaml_scalar(task.name)}',
		f'task: {_yaml_scalar(task.task)}',
		'judge_context:',
	]
	lines.extend(f'  - {_yaml_scalar(item)}' for item in task.judge_context)
	lines.append(f'max_steps: {task.max_steps}')
	path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _load_records(input_path: Path) -> list[Mind2WebRecord]:
	data: Any = json.loads(input_path.read_text(encoding='utf-8'))
	if not isinstance(data, list):
		raise ValueError(f'Expected a list of records in {input_path}')
	return [Mind2WebRecord.model_validate(item) for item in data]


def _to_agent_task(record: Mind2WebRecord, max_steps: int) -> AgentTask:
	action_summary = '; '.join(record.action_reprs[:8])
	judge_context = [
		f'The agent should complete this Mind2Web task on the {record.website} website.',
		f'The target domain is {record.domain}' + (f' / {record.subdomain}.' if record.subdomain else '.'),
		'The final response should clearly state whether the requested web task was completed and include any requested result.',
	]
	if action_summary:
		judge_context.append(f'Reference human action trajectory includes: {action_summary}')

	return AgentTask(
		name=f'Mind2Web {record.website} {record.id}',
		task=f'On the {record.website} website, complete this task: {record.confirmed_task}',
		judge_context=judge_context,
		max_steps=max_steps,
	)


def convert_records(input_path: Path, output_dir: Path, limit: int, offset: int, max_steps: int) -> int:
	"""Convert a slice of Mind2Web records into agent task YAML files."""

	records = _load_records(input_path)
	selected_records = records[offset : offset + limit if limit > 0 else None]
	output_dir.mkdir(parents=True, exist_ok=True)

	for index, record in enumerate(selected_records, start=offset):
		task = _to_agent_task(record, max_steps=max_steps)
		filename = f'{index:04d}_{_slugify(record.website)}_{_slugify(record.id, max_length=36)}.yaml'
		_write_agent_task(output_dir / filename, task)

	return len(selected_records)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		'--input',
		type=Path,
		default=Path('tests/mind2web_data/processed.json'),
		help='Path to processed Mind2Web JSON data.',
	)
	parser.add_argument(
		'--output-dir',
		type=Path,
		default=Path('tests/agent_tasks_mind2web_smoke'),
		help='Directory where generated YAML tasks are written.',
	)
	parser.add_argument('--limit', type=int, default=10, help='Number of records to convert. Use 0 for all records.')
	parser.add_argument('--offset', type=int, default=0, help='Starting record offset.')
	parser.add_argument('--max-steps', type=int, default=20, help='Browser Use max_steps for each task.')
	args = parser.parse_args()

	count = convert_records(
		input_path=args.input,
		output_dir=args.output_dir,
		limit=args.limit,
		offset=args.offset,
		max_steps=args.max_steps,
	)
	print(f'Wrote {count} Mind2Web agent tasks to {args.output_dir}')


if __name__ == '__main__':
	main()
