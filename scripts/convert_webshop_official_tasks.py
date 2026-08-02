"""Generate Browser Use YAML tasks for a running WebShop HTML environment."""

from __future__ import annotations

import argparse
from pathlib import Path


def _yaml_scalar(value: str) -> str:
	escaped = value.replace('\\', '\\\\').replace('"', '\\"')
	return f'"{escaped}"'


def _write_task(
	path: Path,
	name: str,
	task: str,
	judge_context: list[str],
	max_steps: int,
	webshop_reward_threshold: float,
) -> None:
	lines = [
		f'name: {_yaml_scalar(name)}',
		f'task: {_yaml_scalar(task)}',
		'judge_context:',
	]
	lines.extend(f'  - {_yaml_scalar(item)}' for item in judge_context)
	lines.append(f'max_steps: {max_steps}')
	lines.append(f'webshop_reward_threshold: {webshop_reward_threshold}')
	path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def convert(output_dir: Path, base_url: str, limit: int, max_steps: int, webshop_reward_threshold: float) -> int:
	output_dir.mkdir(parents=True, exist_ok=True)
	for old_file in output_dir.glob('*.yaml'):
		old_file.unlink()

	for index in range(limit):
		start_url = f'{base_url.rstrip("/")}/browseruse_fixed_{index}'
		task = (
			f'Open {start_url}. Complete the WebShop shopping instruction shown on the page. '
			'Search using the key product words, required attributes, and required option values from the instruction. '
			'On the product page, verify that the title, attributes, and selected options match the instruction before clicking Buy Now. '
			'After checkout, read the WebShop "Your score" reward and include it in your final response.'
		)
		judge_context = [
			'This task uses the WebShop HTML environment with WebShop reward shown on the final checkout page.',
			'Success means the agent reaches the WebShop checkout/done page and the displayed reward is close to 1.0.',
			'If the final reward is below 1.0, judge the task as failed unless the output clearly proves a benchmark issue.',
		]
		_write_task(
			output_dir / f'{index:04d}_webshop_official_fixed_{index}.yaml',
			f'WebShop Official fixed {index}',
			task,
			judge_context,
			max_steps,
			webshop_reward_threshold,
		)
	return limit


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--output-dir', type=Path, required=True)
	parser.add_argument('--base-url', default='http://127.0.0.1:3002')
	parser.add_argument('--limit', type=int, default=1000)
	parser.add_argument('--max-steps', type=int, default=30)
	parser.add_argument('--webshop-reward-threshold', type=float, default=0.99)
	args = parser.parse_args()
	count = convert(args.output_dir, args.base_url, args.limit, args.max_steps, args.webshop_reward_threshold)
	print(f'Wrote {count} WebShop official tasks to {args.output_dir}')


if __name__ == '__main__':
	main()
