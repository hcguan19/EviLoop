"""Convert WebShop human instructions into Browser Use agent task YAML files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class WebShopInstruction(BaseModel):
	"""One WebShop human shopping instruction."""

	asin: str
	instruction: str
	attributes: list[str] = Field(default_factory=list)
	options: list[str] = Field(default_factory=list)
	instruction_attributes: list[str] = Field(default_factory=list)
	instruction_options: list[str] = Field(default_factory=list)


def _slugify(value: str, max_length: int = 80) -> str:
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


def _load_instructions(path: Path) -> list[WebShopInstruction]:
	data: Any = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(data, dict):
		raise ValueError(f'Expected a JSON object keyed by ASIN in {path}')

	instructions: list[WebShopInstruction] = []
	for asin, items in data.items():
		if not isinstance(items, list):
			continue
		for item in items:
			if not isinstance(item, dict):
				continue
			item = {**item, 'asin': item.get('asin') or asin}
			try:
				instructions.append(WebShopInstruction.model_validate(item))
			except Exception:
				continue
	return instructions


def convert(input_path: Path, output_dir: Path, base_url: str, limit: int, max_steps: int) -> int:
	instructions = _load_instructions(input_path)
	selected = instructions[: limit if limit > 0 else None]
	output_dir.mkdir(parents=True, exist_ok=True)
	for old_file in output_dir.glob('*.yaml'):
		old_file.unlink()

	for index, item in enumerate(selected):
		session_id = f'browseruse_fixed_{index}'
		start_url = f'{base_url.rstrip("/")}/{session_id}'
		judge_context = [
			'This task was converted from the WebShop local simulated e-commerce benchmark.',
			f'Target ASIN: {item.asin}.',
			f'Original shopping instruction: {item.instruction}',
			'Success means finding a product that satisfies the shopping instruction, selecting required options, and buying/submitting it in WebShop.',
		]
		if item.instruction_attributes:
			judge_context.append(f'Required attributes mentioned by the instruction: {json.dumps(item.instruction_attributes, ensure_ascii=False)}')
		if item.instruction_options:
			judge_context.append(f'Required options mentioned by the instruction: {json.dumps(item.instruction_options, ensure_ascii=False)}')

		task = (
			f'Open {start_url}. In the local WebShop site, complete this shopping task: '
			f'{item.instruction}. Use the site search, product pages, options, and buy/submit controls as needed.'
		)
		filename = f'{index:04d}_webshop_{_slugify(item.asin, 20)}_{_slugify(item.instruction, 60)}.yaml'
		_write_task(output_dir / filename, f'WebShop {item.asin}', task, judge_context, max_steps)
	return len(selected)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		'--input',
		type=Path,
		default=Path('datasets/repos/webshop/baseline_models/data/items_human_ins.json'),
	)
	parser.add_argument('--output-dir', type=Path, default=Path('tests/agent_tasks_webshop_1000'))
	parser.add_argument('--base-url', default='http://localhost:3000')
	parser.add_argument('--limit', type=int, default=1000, help='Use 0 for all available instructions.')
	parser.add_argument('--max-steps', type=int, default=25)
	args = parser.parse_args()

	count = convert(args.input, args.output_dir, args.base_url, args.limit, args.max_steps)
	print(f'Wrote {count} WebShop agent tasks to {args.output_dir}')


if __name__ == '__main__':
	main()
