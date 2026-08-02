"""Build a public/direct Browser Use evaluation set from local converted tasks."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _copy_group(source_dir: Path, output_dir: Path, prefix: str, start_index: int) -> int:
	count = 0
	for task_file in sorted(source_dir.glob('*.yaml')):
		destination = output_dir / f'{start_index + count:04d}_{prefix}_{task_file.name.split("_", 1)[1]}'
		shutil.copy2(task_file, destination)
		count += 1
	return count


def build(output_dir: Path, mind2web_dir: Path, miniwob_dir: Path) -> dict[str, object]:
	output_dir.mkdir(parents=True, exist_ok=True)
	for old_file in output_dir.glob('*.yaml'):
		old_file.unlink()

	counts: dict[str, int] = {}
	next_index = 0
	counts['mind2web'] = _copy_group(mind2web_dir, output_dir, 'mind2web', next_index)
	next_index += counts['mind2web']
	counts['miniwob'] = _copy_group(miniwob_dir, output_dir, 'miniwob', next_index)

	manifest = {
		'output_dir': str(output_dir),
		'total_selected': sum(counts.values()),
		'counts_by_source': counts,
		'excluded_sources': {
			'webarena': 'requires deployed WebArena sites such as __SHOPPING_ADMIN__',
			'visualwebarena': 'requires deployed VisualWebArena sites such as __CLASSIFIEDS__',
			'browsergym_webarenalite': 'uses WebArena-derived environment tasks',
			'workarena': 'requires ServiceNow WorkArena environment',
			'osworld': 'requires desktop OSWorld environment',
		},
		'note': 'This is the largest non-duplicated local set that can be attempted without deploying private benchmark environments.',
	}
	(output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
	return manifest


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--output-dir', type=Path, required=True)
	parser.add_argument('--mind2web-dir', type=Path, required=True)
	parser.add_argument('--miniwob-dir', type=Path, required=True)
	args = parser.parse_args()
	print(json.dumps(build(args.output_dir, args.mind2web_dir, args.miniwob_dir), indent=2, ensure_ascii=False))


if __name__ == '__main__':
	main()
