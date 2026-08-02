"""Filter generated Browser Use agent task YAMLs by benchmark source.

This keeps local evaluation sets reproducible after converting multiple
benchmarks into a common YAML format.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


DEFAULT_INCLUDE_SOURCES = ('mind2web', 'miniwob')


def source_from_filename(path: Path) -> str:
	"""Return the benchmark source encoded in a generated YAML filename."""
	parts = path.stem.split('_', 2)
	if len(parts) < 2:
		return 'unknown'
	return parts[1]


def filter_tasks(input_dir: Path, output_dir: Path, include_sources: set[str], limit: int | None) -> dict[str, int]:
	output_dir.mkdir(parents=True, exist_ok=True)

	for existing_file in output_dir.glob('*.yaml'):
		existing_file.unlink()

	counts: dict[str, int] = {}
	selected_count = 0
	for task_file in sorted(input_dir.glob('*.yaml')):
		source = source_from_filename(task_file)
		counts[source] = counts.get(source, 0) + 1
		if source not in include_sources:
			continue
		if limit is not None and selected_count >= limit:
			continue

		destination = output_dir / f'{selected_count:04d}_{task_file.name.split("_", 1)[1]}'
		shutil.copy2(task_file, destination)
		selected_count += 1

	manifest = {
		'input_dir': str(input_dir),
		'output_dir': str(output_dir),
		'include_sources': sorted(include_sources),
		'limit': limit,
		'total_input': sum(counts.values()),
		'total_selected': selected_count,
		'input_counts_by_source': dict(sorted(counts.items())),
	}
	(output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
	return manifest


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('input_dir', type=Path)
	parser.add_argument('output_dir', type=Path)
	parser.add_argument('--include-source', action='append', dest='include_sources')
	parser.add_argument('--limit', type=int)
	args = parser.parse_args()

	include_sources = set(args.include_sources or DEFAULT_INCLUDE_SOURCES)
	manifest = filter_tasks(args.input_dir, args.output_dir, include_sources, args.limit)
	print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == '__main__':
	main()
