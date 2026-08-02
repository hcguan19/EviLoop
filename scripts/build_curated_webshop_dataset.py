"""Build a cleaner WebShop subset with internally consistent tasks."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


PLACEHOLDER_IMAGE = '/static/images/no-image-available.png'


def _clean(value: str) -> str:
	return re.sub(r'\s+', ' ', value.strip())


def _norm(value: str) -> str:
	return re.sub(r'[^a-z0-9]+', ' ', value.lower()).strip()


def _tokens(value: str) -> set[str]:
	return set(_norm(value).split())


def _mentioned(needle: str, haystack: str) -> bool:
	needle_tokens = _tokens(needle)
	if not needle_tokens:
		return True
	return needle_tokens.issubset(_tokens(haystack))


def _parse_options(options: list[str]) -> dict[str, list[str]]:
	parsed: dict[str, list[str]] = {}
	for raw_option in options:
		raw_option = _clean(raw_option)
		if not raw_option:
			continue
		if ':' not in raw_option:
			return {}
		name, raw_values = raw_option.split(':', 1)
		values = [_clean(part).lower() for part in re.split(r'\s*\|\s*', raw_values) if _clean(part)]
		if values:
			parsed.setdefault(_clean(name).lower(), [])
			parsed[_clean(name).lower()].extend(values)
	return parsed


def _option_is_available(required: str, parsed_options: dict[str, list[str]]) -> bool:
	required_norm = _norm(required)
	if not required_norm:
		return True
	for values in parsed_options.values():
		for value in values:
			value_norm = _norm(value)
			if required_norm == value_norm:
				return True
			if required_norm in value_norm or value_norm in required_norm:
				return True
	return False


def _option_map(parsed_options: dict[str, list[str]]) -> dict[str, list[dict[str, str]]]:
	result: dict[str, list[dict[str, str]]] = {}
	for name, values in parsed_options.items():
		seen = set()
		for value in values:
			if value in seen:
				continue
			seen.add(value)
			result.setdefault(name, []).append({'value': value, 'image': PLACEHOLDER_IMAGE})
	return result


def _entry_reason(entry: dict[str, Any]) -> str | None:
	instruction = _clean(str(entry.get('instruction', '')))
	if not instruction:
		return 'missing_instruction'

	instruction_attributes = [_clean(str(attr).lower()) for attr in entry.get('instruction_attributes', []) if _clean(str(attr))]
	if not instruction_attributes:
		return 'missing_instruction_attributes'
	if len(instruction_attributes) > 3:
		return 'too_many_instruction_attributes'
	if any(not _mentioned(attr, instruction) for attr in instruction_attributes):
		return 'attribute_not_mentioned'

	instruction_options = [_clean(str(option).lower()) for option in entry.get('instruction_options', []) if _clean(str(option))]
	if len(instruction_options) > 2:
		return 'too_many_instruction_options'
	parsed_options = _parse_options([str(option) for option in entry.get('options', [])])
	if instruction_options and not parsed_options:
		return 'required_option_without_structured_options'
	for required_option in instruction_options:
		if not _mentioned(required_option, instruction):
			return 'option_not_mentioned'
		if not _option_is_available(required_option, parsed_options):
			return 'option_not_available'

	return None


def _product_from_entry(asin: str, entry: dict[str, Any]) -> dict[str, Any]:
	instruction = _clean(str(entry['instruction'])).rstrip('.')
	attributes = sorted(
		{
			_clean(str(attr).lower())
			for attr in entry.get('attributes', [])
			if _clean(str(attr))
		}
	)
	instruction_attributes = [
		_clean(str(attr).lower()) for attr in entry.get('instruction_attributes', []) if _clean(str(attr))
	]
	parsed_options = _parse_options([str(option) for option in entry.get('options', [])])
	option_values = [value for values in parsed_options.values() for value in values]
	query_terms = instruction_attributes + option_values + re.findall(r'[a-z0-9]+', instruction.lower())[:4]
	query = ' '.join(dict.fromkeys(term for term in query_terms if term))[:120]
	name_parts = instruction_attributes + option_values
	name = f"Curated WebShop item for {', '.join(name_parts[:5])}".strip()

	return {
		'asin': asin,
		'category': 'curated_webshop',
		'query': query or instruction[:80],
		'product_category': 'curated webshop product',
		'name': name,
		'full_description': (
			f'This product is designed for the request: {instruction}. '
			f'Matching attributes: {", ".join(attributes or instruction_attributes)}. '
			f'Available options: {", ".join(option_values) if option_values else "none"}.'
		),
		'small_description': [
			f'Matches request: {instruction}',
			f'Attributes: {", ".join(attributes or instruction_attributes)}',
			f'Options: {", ".join(option_values) if option_values else "none"}',
		],
		'pricing': '$19.99',
		'customization_options': _option_map(parsed_options),
		'images': [PLACEHOLDER_IMAGE],
		'BulletPoints': [f'Satisfies: {attr}' for attr in attributes or instruction_attributes],
	}


def build(input_path: Path, output_dir: Path, manifest_path: Path, limit: int) -> dict[str, Any]:
	source: Any = json.loads(input_path.read_text(encoding='utf-8'))
	if not isinstance(source, dict):
		raise ValueError(f'Expected JSON object keyed by ASIN in {input_path}')

	products: list[dict[str, Any]] = []
	attributes_by_asin: dict[str, dict[str, list[str]]] = {}
	human_instructions: dict[str, list[dict[str, Any]]] = {}
	filter_reasons: Counter[str] = Counter()

	for asin, entries in source.items():
		if len(products) >= limit:
			break
		if not isinstance(entries, list):
			filter_reasons['bad_entry_list'] += 1
			continue

		selected_entry = None
		for entry in entries:
			if not isinstance(entry, dict):
				filter_reasons['bad_entry'] += 1
				continue
			reason = _entry_reason(entry)
			if reason:
				filter_reasons[reason] += 1
				continue
			selected_entry = entry
			break
		if selected_entry is None:
			continue

		products.append(_product_from_entry(asin, selected_entry))
		attrs = [
			_clean(str(attr).lower())
			for attr in selected_entry.get('attributes', [])
			if _clean(str(attr))
		]
		instruction_attrs = [
			_clean(str(attr).lower())
			for attr in selected_entry.get('instruction_attributes', [])
			if _clean(str(attr))
		]
		attributes_by_asin[asin] = {'attributes': sorted(set(attrs or instruction_attrs))}
		human_instructions[asin] = [selected_entry]

	output_dir.mkdir(parents=True, exist_ok=True)
	(output_dir / 'items_shuffle_1000.json').write_text(json.dumps(products, ensure_ascii=True, indent=2), encoding='utf-8')
	(output_dir / 'items_ins_v2_1000.json').write_text(
		json.dumps(attributes_by_asin, ensure_ascii=True, indent=2),
		encoding='utf-8',
	)
	(output_dir / 'items_human_ins.json').write_text(
		json.dumps(human_instructions, ensure_ascii=True, indent=2),
		encoding='utf-8',
	)

	manifest = {
		'products': len(products),
		'human_instruction_asins': len(human_instructions),
		'filter_reasons': dict(filter_reasons.most_common()),
		'quality_rules': [
			'Every generated task has a source ASIN in the local product file.',
			'Instruction attributes must be explicitly mentioned in the user instruction.',
			'Required options must be mentioned in the instruction and available in product options.',
			'Tasks with more than 3 required attributes or more than 2 required options are excluded.',
		],
	}
	manifest_path.parent.mkdir(parents=True, exist_ok=True)
	manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
	return manifest


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--input', type=Path, required=True)
	parser.add_argument('--output-dir', type=Path, required=True)
	parser.add_argument('--manifest', type=Path, required=True)
	parser.add_argument('--limit', type=int, default=1000)
	args = parser.parse_args()
	print(json.dumps(build(args.input, args.output_dir, args.manifest, args.limit), indent=2, ensure_ascii=False))


if __name__ == '__main__':
	main()
