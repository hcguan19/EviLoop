"""Convert downloaded benchmark task definitions into Browser Use agent task YAML files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _slugify(value: str, max_length: int = 90) -> str:
	slug = re.sub(r'[^a-zA-Z0-9]+', '_', value.strip().lower()).strip('_')
	return slug[:max_length] or 'task'


def _yaml_scalar(value: str) -> str:
	escaped = value.replace('\\', '\\\\').replace('"', '\\"')
	return f'"{escaped}"'


def _write_task(path: Path, name: str, task: str, judge_context: list[str], max_steps: int) -> None:
	lines = [
		f'name: {_yaml_scalar(name)}',
		f'task: {_yaml_scalar(task)}',
		'judge_context:',
	]
	lines.extend(f'  - {_yaml_scalar(item)}' for item in judge_context)
	lines.append(f'max_steps: {max_steps}')
	path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _json_items(path: Path) -> list[dict[str, Any]]:
	data = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(data, list):
		raise ValueError(f'Expected list in {path}')
	return [item for item in data if isinstance(item, dict)]


def _reference_summary(item: dict[str, Any]) -> str | None:
	eval_data = item.get('eval')
	if not isinstance(eval_data, dict):
		return None
	reference_answers = eval_data.get('reference_answers')
	if reference_answers:
		return f'Reference answer constraints: {json.dumps(reference_answers, ensure_ascii=False)}'
	raw_annotation = eval_data.get('reference_answer_raw_annotation')
	if raw_annotation:
		return f'Reference answer: {raw_annotation}'
	required = []
	for program in eval_data.get('program_html') or []:
		if isinstance(program, dict) and program.get('required_contents'):
			required.append(program['required_contents'])
	if required:
		return f'Required page contents: {json.dumps(required, ensure_ascii=False)}'
	return None


def _webarena_tasks(dataset: str, paths: Iterable[Path], max_steps: int) -> list[tuple[str, str, list[str], int]]:
	tasks = []
	for path in paths:
		for item in _json_items(path):
			task_id = item.get('task_id', len(tasks))
			intent = str(item.get('intent') or item.get('intent_template') or '').strip()
			start_url = str(item.get('start_url') or '').strip()
			sites = ', '.join(str(site) for site in item.get('sites') or [])
			task_text = f'Complete this {dataset} browser task'
			if sites:
				task_text += f' for site(s) {sites}'
			if start_url:
				task_text += f' starting from {start_url}'
			task_text += f': {intent}'
			judge_context = [
				f'This task was converted from {dataset}.',
				'If the original benchmark environment is not deployed, report the blocker clearly instead of guessing.',
			]
			ref = _reference_summary(item)
			if ref:
				judge_context.append(ref)
			tasks.append((f'{dataset} {task_id}', task_text, judge_context, max_steps))
	return tasks


def _mind2web_tasks(path: Path, max_steps: int) -> list[tuple[str, str, list[str], int]]:
	tasks = []
	for item in _json_items(path):
		task_id = item.get('id', len(tasks))
		website = str(item.get('website') or 'unknown')
		confirmed_task = str(item.get('confirmed_task') or '').strip()
		action_reprs = item.get('action_reprs') or []
		judge_context = [
			f'This task was converted from Mind2Web on the {website} website.',
			f'Domain: {item.get("domain", "unknown")} / {item.get("subdomain", "unknown")}.',
			'The final response should clearly state whether the web task was completed and include any requested result.',
		]
		if action_reprs:
			judge_context.append(f'Reference human action trajectory includes: {"; ".join(map(str, action_reprs[:8]))}')
		tasks.append(
			(
				f'Mind2Web {website} {task_id}',
				f'On the {website} website, complete this task: {confirmed_task}',
				judge_context,
				max_steps,
			)
		)
	return tasks


def _osworld_tasks(path: Path, max_steps: int) -> list[tuple[str, str, list[str], int]]:
	tasks = []
	for file_path in sorted(path.rglob('*.json')):
		item = json.loads(file_path.read_text(encoding='utf-8'))
		task_id = item.get('id') or file_path.stem
		instruction = str(item.get('instruction') or '').strip()
		snapshot = str(item.get('snapshot') or 'unknown')
		related_apps = ', '.join(str(app) for app in item.get('related_apps') or [])
		judge_context = [
			'This task was converted from OSWorld.',
			f'Original snapshot/app context: {snapshot}; related apps: {related_apps}.',
			'Browser Use can only operate in the browser here; report clearly if the task requires unsupported desktop control.',
		]
		tasks.append((f'OSWorld {snapshot} {task_id}', instruction, judge_context, max_steps))
	return tasks


def _miniwob_tasks(path: Path, max_steps: int) -> list[tuple[str, str, list[str], int]]:
	tasks = []
	for file_path in sorted(path.glob('*.html')):
		task_name = file_path.stem
		task_uri = file_path.resolve().as_uri()
		judge_context = [
			'This task was converted from MiniWoB++.',
			'The task instruction is displayed inside the local MiniWoB page.',
			'Success means completing the interaction requested by the page.',
		]
		tasks.append(
			(
				f'MiniWoB++ {task_name}',
				f'Open {task_uri} and complete the MiniWoB++ task shown on the page.',
				judge_context,
				max_steps,
			)
		)
	return tasks


def _workarena_tasks(path: Path, max_steps: int) -> list[tuple[str, str, list[str], int]]:
	tasks = []
	with path.open(encoding='utf-8', newline='') as file:
		for row in csv.DictReader(file):
			task_name = row.get('task_name') or f'workarena_{len(tasks)}'
			category = row.get('category') or 'unknown'
			level = row.get('level') or 'unknown'
			split = row.get('browsergym_split') or 'unknown'
			judge_context = [
				'This task was converted from BrowserGym WorkArena metadata.',
				'The ServiceNow WorkArena environment is not deployed by this converted YAML task.',
				f'Category: {category}; level: {level}; split: {split}.',
			]
			tasks.append(
				(
					f'WorkArena {task_name}',
					f'Complete the WorkArena task named {task_name}. If the ServiceNow environment is unavailable, report that blocker clearly.',
					judge_context,
					max_steps,
				)
			)
	return tasks


def _round_robin(groups: list[list[tuple[str, str, list[str], int]]], limit: int) -> list[tuple[str, str, list[str], int]]:
	selected = []
	index = 0
	while len(selected) < limit:
		progress = False
		for group in groups:
			if index < len(group):
				selected.append(group[index])
				progress = True
				if len(selected) >= limit:
					break
		if not progress:
			break
		index += 1
	return selected


def convert(root: Path, output_dir: Path, limit: int, max_steps: int) -> int:
	datasets = root / 'datasets' / 'repos'
	groups = [
		_mind2web_tasks(root / 'tests' / 'mind2web_data' / 'processed.json', max_steps),
		_webarena_tasks('WebArena', [datasets / 'webarena' / 'config_files' / 'test.raw.json'], max_steps),
		_webarena_tasks('VisualWebArena', sorted((datasets / 'visualwebarena' / 'config_files').rglob('*.raw.json')), max_steps),
		_webarena_tasks(
			'BrowserGym WebArenaLite',
			[datasets / 'browsergym' / 'browsergym' / 'webarenalite' / 'src' / 'browsergym' / 'webarenalite' / 'test_webarena_lite.raw.json'],
			max_steps,
		),
		_osworld_tasks(datasets / 'osworld' / 'evaluation_examples' / 'examples', max_steps),
		_miniwob_tasks(datasets / 'miniwob' / 'miniwob' / 'html' / 'miniwob', max_steps),
		_workarena_tasks(
			datasets
			/ 'browsergym'
			/ 'browsergym'
			/ 'experiments'
			/ 'src'
			/ 'browsergym'
			/ 'experiments'
			/ 'benchmark'
			/ 'metadata'
			/ 'workarena.csv',
			max_steps,
		),
	]
	selected = _round_robin(groups, limit=limit)
	output_dir.mkdir(parents=True, exist_ok=True)
	for old_file in output_dir.glob('*.yaml'):
		old_file.unlink()
	for index, (name, task, judge_context, task_max_steps) in enumerate(selected):
		filename = f'{index:04d}_{_slugify(name)}.yaml'
		_write_task(output_dir / filename, name, task, judge_context, task_max_steps)
	return len(selected)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--root', type=Path, default=Path.cwd(), help='Browser Use project root.')
	parser.add_argument('--output-dir', type=Path, default=Path('tests/agent_tasks_downloaded_2000'))
	parser.add_argument('--limit', type=int, default=2000)
	parser.add_argument('--max-steps', type=int, default=20)
	args = parser.parse_args()
	count = convert(args.root, args.output_dir, args.limit, args.max_steps)
	print(f'Wrote {count} converted benchmark tasks to {args.output_dir}')


if __name__ == '__main__':
	main()
