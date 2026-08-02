"""Build WebShop-compatible product files from bundled human instructions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PLACEHOLDER_IMAGE = '/static/images/no-image-available.png'


def _clean(value: str) -> str:
	return re.sub(r'\s+', ' ', value.strip())


def _option_map(options: list[str]) -> dict[str, list[dict[str, str]]]:
	result: dict[str, list[dict[str, str]]] = {}
	for option in options:
		if not option:
			continue
		if ':' in option:
			name, value = option.split(':', 1)
		else:
			name, value = 'option', option
		name = _clean(name).lower()
		values = [_clean(part).lower() for part in re.split(r'\s*\|\s*', value) if _clean(part)]
		if values:
			result.setdefault(name, [])
			for value_item in values:
				result[name].append({'value': value_item, 'image': PLACEHOLDER_IMAGE})
	return result


def build(input_path: Path, output_dir: Path, limit: int) -> dict[str, int]:
	source: Any = json.loads(input_path.read_text(encoding='utf-8'))
	if not isinstance(source, dict):
		raise ValueError(f'Expected JSON object keyed by ASIN in {input_path}')

	products = []
	attributes_by_asin: dict[str, dict[str, list[str]]] = {}
	human_instructions: dict[str, list[dict[str, Any]]] = {}

	for asin, entries in source.items():
		if len(products) >= limit:
			break
		if not isinstance(entries, list) or not entries:
			continue
		valid_entries = [entry for entry in entries if isinstance(entry, dict) and entry.get('instruction')]
		if not valid_entries:
			continue
		first = valid_entries[0]
		instruction = _clean(str(first['instruction']))
		attrs = sorted({str(attr).lower() for entry in valid_entries for attr in entry.get('attributes', [])})
		instruction_attrs = sorted(
			{str(attr).lower() for entry in valid_entries for attr in entry.get('instruction_attributes', [])}
		)
		options = _option_map([str(option) for entry in valid_entries for option in entry.get('options', [])])
		query_terms = instruction_attrs or attrs or re.findall(r'[a-z0-9]+', instruction.lower())[:4]
		query = ' '.join(query_terms[:5]) or instruction[:60]
		product_name = instruction.rstrip('.').capitalize()

		products.append(
			{
				'asin': asin,
				'category': 'webshop',
				'query': query,
				'product_category': 'webshop product',
				'name': product_name,
				'full_description': f'{product_name}. Attributes: {", ".join(attrs) or "general product"}.',
				'small_description': [f'Satisfies: {attr}' for attr in attrs] or [product_name],
				'pricing': '$19.99',
				'customization_options': options,
				'images': [PLACEHOLDER_IMAGE],
				'BulletPoints': [f'Satisfies: {attr}' for attr in attrs] or [product_name],
			}
		)
		attributes_by_asin[asin] = {'attributes': attrs or instruction_attrs or ['general product']}
		human_instructions[asin] = valid_entries

	output_dir.mkdir(parents=True, exist_ok=True)
	(output_dir / 'items_shuffle_1000.json').write_text(json.dumps(products, ensure_ascii=True, indent=2), encoding='utf-8')
	(output_dir / 'items_ins_v2_1000.json').write_text(
		json.dumps(attributes_by_asin, ensure_ascii=True, indent=2), encoding='utf-8'
	)
	(output_dir / 'items_human_ins.json').write_text(
		json.dumps(human_instructions, ensure_ascii=True, indent=2), encoding='utf-8'
	)
	return {'products': len(products), 'human_instruction_asins': len(human_instructions)}


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--input', type=Path, required=True)
	parser.add_argument('--output-dir', type=Path, required=True)
	parser.add_argument('--limit', type=int, default=1000)
	args = parser.parse_args()
	print(json.dumps(build(args.input, args.output_dir, args.limit), indent=2, ensure_ascii=False))


if __name__ == '__main__':
	main()
