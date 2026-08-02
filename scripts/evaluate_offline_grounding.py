"""Run the shared visible-candidate ranker on official offline web benchmarks."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from lxml import html
from pydantic import BaseModel, Field

from browser_use.general_policy.grounding import (
	GroundingExample,
	GroundingPrediction,
	prediction_from_ranked,
	rank_grounding_example,
	summarize_grounding_predictions,
)
from browser_use.general_policy.semantic_reranker import SemanticRerankConfig, semantic_rerank
from browser_use.llm.openai.chat import ChatOpenAI
from browser_use.tools.task_policy import PageCapabilities, TaskObservation, VisibleCandidate


class EvaluationConfig(BaseModel):
	"""Validated offline evaluation configuration."""

	dataset: Literal['mind2web', 'weblinx']
	limit: int = Field(default=20, ge=1, le=1000)
	seed: int = 17
	output: Path
	hf_home: Path
	semantic_rerank: bool = False
	rerank_top_k: int = Field(default=5, ge=2, le=10)
	rerank_margin: float = Field(default=1.0, ge=0.0, le=10.0)


def _stable_identifier(value: str) -> str:
	return hashlib.sha1(value.encode('utf-8')).hexdigest()[:16]


def _attributes(value: Any) -> dict[str, str]:
	if isinstance(value, dict):
		return {str(key): str(item) for key, item in value.items()}
	try:
		parsed = json.loads(str(value or '{}'))
	except json.JSONDecodeError:
		return {}
	return {str(key): str(item) for key, item in parsed.items()} if isinstance(parsed, dict) else {}


def _compact_text(value: str, limit: int = 1600) -> str:
	return re.sub(r'\s+', ' ', value).strip()[:limit]


def _mind2web_candidate(candidate: dict[str, Any], dom_by_id: dict[str, Any]) -> VisibleCandidate:
	attributes = _attributes(candidate.get('attributes'))
	identifier = str(candidate.get('backend_node_id') or attributes.get('backend_node_id') or '')
	node = dom_by_id.get(identifier)
	local_text = ''
	parent_text = ''
	if node is not None:
		local_text = _compact_text(' '.join(node.itertext()), 500)
		if node.getparent() is not None:
			parent_text = _compact_text(' '.join(node.getparent().itertext()), 900)
	semantic_attributes = ' '.join(
		attributes.get(name, '') for name in ('aria_label', 'aria-label', 'name', 'placeholder', 'title', 'value', 'id', 'class')
	)
	label = _compact_text(f'{candidate.get("tag", "")} {semantic_attributes} {local_text}', 500)
	return VisibleCandidate(
		identifier=identifier,
		label=label or str(candidate.get('tag') or identifier),
		visible_text=_compact_text(f'{label} {parent_text}'),
		metadata={'tag': str(candidate.get('tag') or ''), 'attributes': json.dumps(attributes, ensure_ascii=False)},
	)


def _shuffled_candidates(
	positive: list[VisibleCandidate],
	negative: list[VisibleCandidate],
	*,
	seed: int,
	example_id: str,
	max_candidates: int = 100,
) -> tuple[list[VisibleCandidate], set[str]]:
	"""Bound and shuffle candidates deterministically while retaining every positive."""

	rng = random.Random(f'{seed}:{example_id}')
	negative = list(negative)
	rng.shuffle(negative)
	combined = [*positive, *negative[: max(0, max_candidates - len(positive))]]
	rng.shuffle(combined)
	return combined, {candidate.identifier for candidate in positive}


def iter_mind2web_examples(limit: int, seed: int) -> Iterator[GroundingExample]:
	"""Map first-step official DOM observations into the shared grounding contract."""

	from datasets import load_dataset

	dataset = load_dataset('osunlp/Mind2Web', split='train', streaming=True)
	for row_index, row in enumerate(dataset):
		if row_index >= limit:
			break
		actions = row.get('actions') or []
		if not actions:
			continue
		action = actions[0]
		positive = action.get('pos_candidates') or []
		negative = action.get('neg_candidates') or []
		if not positive:
			continue
		try:
			document = html.fromstring(action.get('cleaned_html') or action.get('raw_html') or '<html/>')
		except (ValueError, TypeError):
			continue
		dom_by_id = {
			str(node.get('backend_node_id')): node
			for node in document.xpath('//*[@backend_node_id]')
			if node.get('backend_node_id')
		}
		positive_candidates = [_mind2web_candidate(candidate, dom_by_id) for candidate in positive]
		negative_candidates = [_mind2web_candidate(candidate, dom_by_id) for candidate in negative]
		example_id = f'{row.get("annotation_id", row_index)}:0'
		candidates, positives = _shuffled_candidates(
			positive_candidates,
			negative_candidates,
			seed=seed,
			example_id=example_id,
		)
		yield GroundingExample(
			example_id=example_id,
			objective=str(row.get('confirmed_task') or ''),
			operation=str((action.get('operation') or {}).get('op') or ''),
			positive_identifiers=positives,
			observation=TaskObservation(
				modality='offline_trajectory',
				visible_text=_compact_text(' '.join(document.itertext()), 4000),
				capabilities=PageCapabilities(candidate_count=len(candidates)),
				candidates=candidates,
				action_history_available=False,
				provenance='recorded_trajectory',
			),
		)


def _weblinx_candidate(value: str, index: int) -> VisibleCandidate:
	xpath_match = re.search(r'\[\[xpath\]\]\s*(.+)', value)
	text_match = re.search(r'\[\[text\]\]\s*(.*)', value)
	tag_match = re.search(r'\[\[tag\]\]\s*(\S+)', value)
	attributes_match = re.search(r'\[\[attributes\]\]\s*([^\r\n]*)', value)
	attributes = attributes_match.group(1).strip() if attributes_match else ''
	identifier = (xpath_match.group(1).strip() if xpath_match else '') or _stable_identifier(value)
	label = ' '.join(
		part for part in (tag_match.group(1) if tag_match else '', text_match.group(1) if text_match else '') if part
	)
	return VisibleCandidate(
		identifier=identifier,
		label=_compact_text(label or f'candidate {index}', 500),
		visible_text=_compact_text(value),
		metadata={'tag': tag_match.group(1) if tag_match else '', 'attributes': attributes},
	)


def iter_weblinx_examples(limit: int, seed: int) -> Iterator[GroundingExample]:
	"""Map the official held-out reranking split into the shared grounding contract."""

	from datasets import load_dataset

	dataset = load_dataset('McGill-NLP/weblinx', 'reranking', split='test', streaming=True)
	for row_index, row in enumerate(dataset):
		if row_index >= limit:
			break
		positive_text = list(row.get('positive') or [])
		candidate_texts = [*positive_text, *(row.get('negative') or [])]
		if not positive_text:
			continue
		positive_candidates = [_weblinx_candidate(value, index) for index, value in enumerate(positive_text)]
		negative_candidates = [
			_weblinx_candidate(value, index + len(positive_text)) for index, value in enumerate(row.get('negative') or [])
		]
		example_id = str(row.get('query_id') or row_index)
		candidates, positives = _shuffled_candidates(
			positive_candidates,
			negative_candidates,
			seed=seed,
			example_id=example_id,
		)
		yield GroundingExample(
			example_id=example_id,
			objective=str(row.get('query') or ''),
			positive_identifiers=positives,
			observation=TaskObservation(
				modality='offline_trajectory',
				capabilities=PageCapabilities(candidate_count=len(candidates)),
				candidates=candidates,
				action_history_available=True,
				provenance='recorded_trajectory',
			),
		)


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open('w', encoding='utf-8') as handle:
		for record in records:
			handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def _build_semantic_llm() -> ChatOpenAI:
	api_key = (
		os.getenv('QWEN_CHAT_API_KEY')
		or os.getenv('QWEN_API_KEY')
		or os.getenv('QWEN_EMBED_API_KEY')
		or os.getenv('DASHSCOPE_API_KEY')
	)
	base_url = os.getenv('QWEN_CHAT_BASE_URL') or os.getenv('QWEN_EMBED_BASE_URL')
	model = os.getenv('QWEN_CHAT_MODEL')
	if not (api_key and base_url and model):
		raise RuntimeError('Semantic reranking requires Qwen API key, base URL, and QWEN_CHAT_MODEL.')
	return ChatOpenAI(
		model=model,
		api_key=api_key,
		base_url=base_url,
		temperature=0,
		frequency_penalty=None,
		max_completion_tokens=300,
		add_schema_to_system_prompt=True,
		dont_force_structured_output=True,
	)


async def _evaluate(config: EvaluationConfig) -> dict[str, Any]:
	rerank_config = SemanticRerankConfig(
		enabled=config.semantic_rerank,
		top_k=config.rerank_top_k,
		margin_threshold=config.rerank_margin,
	)
	llm = _build_semantic_llm() if config.semantic_rerank else None
	examples = (
		iter_mind2web_examples(config.limit, config.seed)
		if config.dataset == 'mind2web'
		else iter_weblinx_examples(config.limit, config.seed)
	)
	predictions: list[GroundingPrediction] = []
	records: list[dict[str, Any]] = []
	decisions = []
	for example in examples:
		base_prediction, ranked = rank_grounding_example(example)
		decision = None
		if llm is not None:
			reranked, decision = await semantic_rerank(
				objective=base_prediction.focused_objective,
				operation=example.operation,
				ranked=ranked,
				llm=llm,
				config=rerank_config,
			)
			prediction = prediction_from_ranked(example, base_prediction.focused_objective, reranked)
			decisions.append(decision)
		else:
			prediction = base_prediction
		predictions.append(prediction)
		record = prediction.model_dump(mode='json')
		record['semantic_rerank'] = decision.model_dump(mode='json') if decision is not None else None
		records.append(record)

	summary = {
		'dataset': config.dataset,
		'split': 'train_first_action_diagnostic' if config.dataset == 'mind2web' else 'test_reranking',
		'seed': config.seed,
		'method': 'confidence_gated_semantic_reranker' if config.semantic_rerank else 'shared_visible_candidate_ranker',
		'observation_provenance': 'official_recorded_visible_state',
		'scope': 'offline_element_grounding_not_end_to_end_task_success',
		**summarize_grounding_predictions(predictions),
		'semantic_rerank_enabled': config.semantic_rerank,
		'semantic_rerank_triggered': sum(decision.triggered for decision in decisions),
		'semantic_rerank_errors': sum(decision.error is not None for decision in decisions),
		'prompt_tokens': sum(decision.prompt_tokens or 0 for decision in decisions),
		'completion_tokens': sum(decision.completion_tokens or 0 for decision in decisions),
		'total_tokens': sum(decision.total_tokens or 0 for decision in decisions),
	}
	_write_jsonl(config.output, records)
	config.output.with_suffix('.summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
	print(json.dumps(summary, ensure_ascii=False, indent=2))
	return summary


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('dataset', choices=('mind2web', 'weblinx'))
	parser.add_argument('--limit', type=int, default=20)
	parser.add_argument('--seed', type=int, default=17)
	parser.add_argument('--output', type=Path, required=True)
	parser.add_argument('--hf-home', type=Path, default=Path(os.getenv('HF_HOME', '.cache/huggingface')))
	parser.add_argument('--semantic-rerank', action='store_true')
	parser.add_argument('--rerank-top-k', type=int, default=5)
	parser.add_argument('--rerank-margin', type=float, default=1.0)
	args = parser.parse_args()
	load_dotenv(Path.cwd() / '.env')
	config = EvaluationConfig(**vars(args))
	os.environ['HF_HOME'] = str(config.hf_home)
	os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

	asyncio.run(_evaluate(config))


if __name__ == '__main__':
	main()
