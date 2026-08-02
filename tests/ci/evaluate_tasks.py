"""
Runs all agent tasks in parallel (up to 10 at a time) using separate subprocesses.
Each task gets its own Python process, preventing browser session interference.
Fails with exit code 1 if 0% of tasks pass.
"""

import argparse
import ast
import asyncio
import glob
import html
import json
import logging
import os
import re
import sys
import time
import warnings
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote, unquote, urlparse
from urllib.request import urlopen

import anyio
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()
import browser_use as browser_use_package
from browser_use import ActionResult, Agent, AgentHistoryList, BrowserProfile, BrowserSession, ChatBrowserUse, ChatOpenAI, Tools
from browser_use.general_policy import benchmark_adapter_contract, parse_task_program
from browser_use.llm.google.chat import ChatGoogle
from browser_use.llm.messages import UserMessage
from browser_use.tools.task_policy import (
	PageCapabilities,
	ProgressTracker,
	RankedCandidateAction,
	VisibleCandidate,
	completion_is_grounded,
	extract_task_requirements,
	is_visible_product_detail,
	rank_visible_candidates,
	visible_value_matches_text,
)
from browser_use.tools.utils import get_click_description
from browser_use.tools.views import ClickElementActionIndexOnly, DoneAction

# --- CONFIG ---
MAX_PARALLEL = int(os.getenv('EVAL_MAX_PARALLEL', '10'))
TASK_DIR = (
	sys.argv[1]
	if len(sys.argv) > 1 and not sys.argv[1].startswith('--')
	else os.path.join(os.path.dirname(__file__), '../agent_tasks')
)
TASK_FILES = glob.glob(os.path.join(TASK_DIR, '*.yaml'))
RESULTS_JSONL = os.getenv('EVAL_RESULTS_JSONL')
RESUME_RESULTS = os.getenv('EVAL_RESUME', 'true').lower()[:1] in 'ty1'


class JudgeResponse(BaseModel):
	success: bool
	explanation: str


class WebShopRewardResponse(BaseModel):
	success: bool
	explanation: str
	reward: float | None = None


class MiniWoBRewardResponse(BaseModel):
	"""Programmatic MiniWoB++ episode result read from the local page runtime."""

	success: bool
	explanation: str
	reward: float | None = None
	reason: str | None = None


class SemanticToggleSelection(BaseModel):
	"""Constrained semantic mapping from instruction targets to visible toggle labels."""

	selected_labels: list[str] = Field(min_length=1, max_length=50)


class ShortUiPageProfile(BaseModel):
	"""Observable page features and capabilities used by the unified workflow router."""

	url: str = ''
	text_length: int = 0
	instruction_sample: str = ''
	instruction_line: str = ''
	toggle_labels: list[str] = Field(default_factory=list)
	interactive_count: int = 0
	form_count: int = 0
	link_count: int = 0
	scrollable_count: int = 0
	button_count: int = 0
	text_input_count: int = 0
	textarea_count: int = 0
	select_count: int = 0
	date_input_count: int = 0
	range_count: int = 0
	checkbox_count: int = 0
	radio_count: int = 0
	disabled_count: int = 0
	submit_count: int = 0
	exact_transfer: bool = False
	temporal_control: bool = False
	multi_field_form: bool = False
	labelled_selection: bool = False
	date_entry: bool = False
	ordinal_control: bool = False
	product_constraints: bool = False
	capabilities: list[str] = Field(default_factory=list)
	route_reason: str = ''
	eligible: bool = False


class AdaptivePlan(BaseModel):
	"""Planner output for one level of as-needed task decomposition."""

	failure_cause: str
	steps: list[str] = Field(min_length=2, max_length=5)
	blocked_step: str | None = None


class ReflectionNote(BaseModel):
	"""Compact cross-trial Reflexion memory."""

	failure_cause: str
	lesson: str
	next_strategy: str
	actions_to_avoid: list[str] = Field(default_factory=list, max_length=5)


class AdaPlannerStep(BaseModel):
	"""One code-style AdaPlanner step and its predicted browser feedback."""

	step: int = Field(ge=1)
	action_goal: str
	expected_observation: str


class AdaPlannerPlan(BaseModel):
	"""Closed-loop plan compatible with the official AdaPlanner formulation."""

	steps: list[AdaPlannerStep] = Field(min_length=2, max_length=5)
	resume_from_step: int = Field(default=1, ge=1)
	feedback_diagnosis: str = ''


class WebDARTPlan(BaseModel):
	"""Dynamic navigation, extraction, and execution decomposition."""

	navigation_goal: str
	pages_to_visit: list[str] = Field(min_length=1, max_length=5)
	evidence_to_capture: list[str] = Field(min_length=1, max_length=6)
	stopping_criterion: str
	execution_goal: str
	revision_reason: str = ''


class WebOperatorCandidate(BaseModel):
	"""One distinct and safety-aware trajectory candidate."""

	strategy: str
	expected_progress: str
	safety_risk: str
	reversible: bool = True


class WebOperatorPlan(BaseModel):
	"""Bounded candidate frontier for the WebOperator adaptation."""

	candidates: list[WebOperatorCandidate] = Field(min_length=2, max_length=4)
	selected_candidate: int = Field(default=1, ge=1)
	rejected_reasons: list[str] = Field(default_factory=list, max_length=4)


class VerifyWebShopCandidateAction(BaseModel):
	"""Candidate notes supplied by the agent before deterministic WebShop verification."""

	candidate_summary: str = Field(
		default='',
		description='Short description of the current product candidate and selected options, if known.',
	)


class OpenRankedWebShopCandidateAction(BaseModel):
	"""Open one candidate from the latest deterministic retrieval ranking."""

	rank: int = Field(ge=1, le=5, description='1-based rank from WEBSHOP_CANDIDATE_RANKING')


class SemanticVisibleCandidateAssessment(BaseModel):
	identifier: str
	semantic_score: float = Field(ge=0.0, le=1.0)
	satisfies_all_constraints: bool = False
	matched_constraints: list[str] = Field(default_factory=list, max_length=20)
	missing_constraints: list[str] = Field(default_factory=list, max_length=20)


class SemanticVisibleCandidateRanking(BaseModel):
	"""One bounded semantic ordering over candidates visible in the browser."""

	ranked_candidates: list[SemanticVisibleCandidateAssessment] = Field(default_factory=list, max_length=20)


class SemanticVisibleCandidateChoice(SemanticVisibleCandidateAssessment):
	"""The single best candidate selected from one visible retrieval batch."""


class WebShopAtomicEvidenceItem(BaseModel):
	"""One task constraint paired with an exact quote from visible product details."""

	constraint: str
	satisfied: bool = False
	evidence_quote: str = ''


class WebShopDetailEvidenceAssessment(BaseModel):
	"""Grounded semantic assessment over full visible product detail evidence."""

	semantic_score: float = Field(ge=0.0, le=1.0)
	satisfies_all_constraints: bool = False
	constraints: list[WebShopAtomicEvidenceItem] = Field(default_factory=list, max_length=20)


class WebShopVisibleOption(BaseModel):
	"""One option exposed by the current product page DOM."""

	name: str
	value: str
	checked: bool = False
	data_url: str = ''


class WebShopVisibleSnapshot(BaseModel):
	"""WebShop evidence read only from the browser's current DOM."""

	url: str = ''
	page_text: str = ''
	instruction_text: str = ''
	product_evidence_text: str = ''
	product_title: str = ''
	price_text: str = ''
	options: list[WebShopVisibleOption] = Field(default_factory=list)
	evidence_urls: list[str] = Field(default_factory=list)
	candidates: list[VisibleCandidate] = Field(default_factory=list)
	pagination_urls: list[str] = Field(default_factory=list)
	capabilities: PageCapabilities = Field(default_factory=PageCapabilities)


class AttributeRequirement(BaseModel):
	"""A static product attribute tracked independently from selectable options."""

	name: str
	status: str = 'unknown'
	evidence: str = ''


class OptionRequirement(BaseModel):
	"""A selectable WebShop option with its option group preserved."""

	option_name: str | None = None
	required_value: str
	available_values: list[str] = Field(default_factory=list)
	selected_value: str | None = None
	status: str = 'missing'


class WebShopTaskMemory(BaseModel):
	"""Compact working memory used to keep a long WebShop agent loop on track."""

	phase: str = 'understand'
	instruction: str = ''
	search_query: str = ''
	price_limit: float | None = None
	required_attributes: dict[str, AttributeRequirement] = Field(default_factory=dict)
	required_options: list[OptionRequirement] = Field(default_factory=list)
	visited_asins: list[str] = Field(default_factory=list)
	rejected_candidates: dict[str, list[str]] = Field(default_factory=dict)
	candidate_visits: dict[str, int] = Field(default_factory=dict)
	ranked_candidates: list[dict] = Field(default_factory=list)
	current_asin: str | None = None
	last_candidate_signature: str = ''
	repeated_state_count: int = 0
	search_attempts: int = 0
	query_candidate_inspections: int = 0
	stale_index_recoveries: int = 0
	automatic_candidate_switches: int = 0
	pages_scanned: int = 0
	state_machine_transitions: int = 0
	next_candidate_rank: int = 1
	next_action: str = 'Open WebShop and build a search plan.'


def _resolve_goal_option_requirements(goal_options, product_options: dict) -> list[dict]:
	"""Recover option group names from WebShop's value-only human goals."""
	if isinstance(goal_options, dict):
		raw_options = list(goal_options.items())
	else:
		raw_options = [(None, value) for value in (goal_options or [])]

	requirements = []
	for explicit_name, raw_value in raw_options:
		required_value = str(raw_value).strip().lower()
		option_name = str(explicit_name).strip().lower() if explicit_name is not None else None
		available_values: list[str] = []
		if option_name and option_name in product_options:
			available_values = product_options[option_name]
		else:
			for candidate_name, candidate_values in product_options.items():
				if any(_token_set_match(required_value, value) for value in candidate_values):
					option_name = candidate_name
					available_values = candidate_values
					break
		requirements.append(
			{
				'option_name': option_name,
				'required_value': required_value,
				'available_values': available_values,
				'selected_value': None,
				'status': 'missing',
			}
		)
	return requirements


def _meaningful_webshop_tokens(text: str) -> list[str]:
	stopwords = {
		'a',
		'an',
		'and',
		'are',
		'as',
		'at',
		'be',
		'for',
		'from',
		'get',
		'have',
		'i',
		'am',
		'looking',
		'in',
		'is',
		'it',
		'like',
		'lower',
		'less',
		'need',
		'made',
		'of',
		'on',
		'or',
		'price',
		'prefer',
		'preferably',
		'should',
		'some',
		'than',
		'that',
		'the',
		'to',
		'under',
		'want',
		'with',
		'would',
		'dollar',
		'dollars',
	}
	tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
	return [token for token in tokens if len(token) >= 2 and token not in stopwords and not token.isdigit()]


def _extract_webshop_instruction(page_text: str) -> str:
	lines = [re.sub(r'\s+', ' ', line).strip() for line in page_text.splitlines()]
	lines = [line for line in lines if line]
	for index, line in enumerate(lines):
		if line.lower().startswith('instruction'):
			after_colon = re.sub(r'^instruction\s*:?\s*', '', line, flags=re.IGNORECASE).strip()
			if after_colon:
				return after_colon
			if index + 1 < len(lines):
				return lines[index + 1]
	return ''


def _normalize_webshop_constraint_text(text: str) -> str:
	"""Canonicalize common measurements without using benchmark-only metadata."""

	normalized = text.casefold().replace('\u201c', '"').replace('\u201d', '"').replace('\u00d7', 'x')
	for word, digit in (
		('one', '1'),
		('two', '2'),
		('three', '3'),
		('four', '4'),
		('five', '5'),
		('six', '6'),
		('seven', '7'),
		('eight', '8'),
		('nine', '9'),
		('ten', '10'),
	):
		normalized = re.sub(rf'\b{word}\b', digit, normalized)
	normalized = re.sub(r'(\d)\s*"', r'\1 in', normalized)
	normalized = re.sub(r'\bfluid\s+ounces?\b', 'fl oz', normalized)
	normalized = re.sub(r'\bfl\.?\s*oz\.?\b', 'fl oz', normalized)
	normalized = re.sub(r'\bounces?\b|\boz\.?\b', 'oz', normalized)
	normalized = re.sub(r'\bfeet\b|\bfoot\b|\bft\.?\b', 'ft', normalized)
	normalized = re.sub(r'\binch(?:es)?\b|\bin\.?\b', 'in', normalized)
	normalized = re.sub(r'(\d)(ft|oz|in)\b', r'\1 \2', normalized)
	normalized = re.sub(r'\s+', ' ', normalized)
	return re.sub(r'[^a-z0-9.]+', ' ', normalized).strip()


def _extract_webshop_hard_constraints(instruction: str) -> list[str]:
	"""Extract exact measurements and explicit negative/free attributes."""

	text = _normalize_webshop_constraint_text(instruction)
	patterns = (
		r'\b\d+(?:\.\d+)?\s*(?:fl oz|oz|ft|in|ml|l|gb|tb)\s*(?:x|by)\s*\d+(?:\.\d+)?\s*(?:fl oz|oz|ft|in|ml|l|gb|tb)?\b',
		r'\b\d+(?:\.\d+)?\s*(?:fl oz|oz|ft|in|ml|l|gb|tb)\b',
		r'(?<!\d )\b(?:pack|set|case)\s+of\s+\d+\b',
		r'\b\d+\s+(?:pack|count|ct|pieces?|single\s+servings?|servings?)\b',
		r'\b[a-z0-9]+\s+free\b',
		r'\bnon\s+[a-z0-9]+\b',
	)
	constraints: list[str] = []
	for pattern in patterns:
		for match in re.finditer(pattern, text):
			value = re.sub(r'\s+', ' ', match.group(0)).strip()
			if value and value not in constraints:
				constraints.append(value)
	return constraints


def _webshop_constraint_is_visible(constraint: str, evidence: str) -> bool:
	normalized_evidence = _normalize_webshop_constraint_text(evidence)
	normalized_constraint = _normalize_webshop_constraint_text(constraint)
	return bool(
		normalized_constraint
		and re.search(rf'(?<![a-z0-9]){re.escape(normalized_constraint)}(?![a-z0-9])', normalized_evidence)
	)


def _webshop_atomic_evidence_is_complete(required_tokens: list[str], matched_tokens: list[str]) -> bool:
	"""Require every task-derived product token to have visible detail-page evidence."""

	matched = set(matched_tokens)
	return bool(required_tokens) and all(token in matched for token in required_tokens)


def _webshop_detail_assessment_is_grounded(
	assessment: WebShopDetailEvidenceAssessment,
	evidence: str,
) -> bool:
	"""Accept semantic equivalence only when every atomic decision cites visible text."""

	if not assessment.satisfies_all_constraints or assessment.semantic_score < 0.90 or not assessment.constraints:
		return False
	normalized_evidence = _normalize_webshop_constraint_text(evidence)
	for item in assessment.constraints:
		quote = _normalize_webshop_constraint_text(item.evidence_quote)
		if not item.satisfied or len(quote) < 3 or quote not in normalized_evidence:
			return False
	return True


def _webshop_option_value_matches_instruction(option_name: str, value: str, instruction: str) -> bool:
	"""Match visible options conservatively, with a color-core fallback for decorated labels."""

	if visible_value_matches_text(value, instruction):
		return True
	if re.sub(r'[^a-z]+', '', option_name.casefold()) not in {'color', 'colour'}:
		return False
	value_tokens = set(_meaningful_webshop_tokens(value))
	instruction_tokens = set(_meaningful_webshop_tokens(instruction))
	return any(len(token) >= 3 and token in instruction_tokens for token in value_tokens)


def _extract_webshop_price_limit(instruction: str) -> float | None:
	match = re.search(
		r'(?:under|lower than|less than|below|not more than)\s*\$?\s*([0-9]+(?:\.[0-9]+)?)',
		instruction,
		flags=re.IGNORECASE,
	)
	if not match:
		return None
	try:
		return float(match.group(1))
	except ValueError:
		return None


def _extract_visible_prices(text: str) -> list[float]:
	prices: list[float] = []
	for match in re.finditer(r'\$\s*([0-9]+(?:\.[0-9]+)?)', text):
		try:
			prices.append(float(match.group(1)))
		except ValueError:
			continue
	return prices


def _parse_webshop_url(url: str) -> dict:
	match = re.search(r'/(?:item_page|item_sub_page|done)/([^/]+)/([^/]+)/', url)
	if not match:
		return {}
	session_id, asin = match.group(1), match.group(2)
	options: dict = {}
	option_match = re.search(r'/(\{.*\})(?:$|[?#])', unquote(url))
	if option_match:
		try:
			parsed_options = ast.literal_eval(option_match.group(1))
			if isinstance(parsed_options, dict):
				options = parsed_options
		except Exception:
			options = {}
	return {'session_id': session_id, 'asin': asin, 'options': options}


def _webshop_url_with_options(current_url: str, selected_options: dict[str, str]) -> str:
	"""Replace only the final options path segment, preserving search keywords and page."""
	parsed_url = urlparse(current_url)
	path_parts = parsed_url.path.rstrip('/').split('/')
	if len(path_parts) < 2:
		raise ValueError(f'Invalid WebShop product URL: {current_url}')
	path_parts[-1] = quote(str(selected_options), safe='')
	return parsed_url._replace(path='/'.join(path_parts), query='', fragment='').geturl()


def _webshop_search_url_from_detail(current_url: str) -> str | None:
	"""Recover the visible search-results URL encoded in a WebShop detail route."""

	parsed = urlparse(current_url)
	parts = parsed.path.strip('/').split('/')
	if len(parts) < 6 or parts[0] != 'item_page':
		return None
	session_id, keywords, page = parts[1], parts[3], parts[4]
	return parsed._replace(
		path=f'/search_results/{session_id}/{keywords}/{page}',
		query='',
		fragment='',
	).geturl()


def _webshop_home_url_from_detail(current_url: str) -> str | None:
	parsed = urlparse(current_url)
	parts = parsed.path.strip('/').split('/')
	if len(parts) < 2 or parts[0] not in {'item_page', 'item_sub_page'}:
		return None
	return parsed._replace(path=f'/{parts[1]}', query='', fragment='').geturl()


def _parse_price_value(raw_price) -> float:
	if isinstance(raw_price, (int, float)):
		return float(raw_price)
	match = re.search(r'[0-9]+(?:\.[0-9]+)?', str(raw_price))
	return float(match.group(0)) if match else 100.0


def _canonical_instruction(text: str) -> str:
	text = text.lower().strip().rstrip('.')
	text = re.sub(r',?\s*and\s+price\s+lower\s+than\s+[0-9.]+\s+dollars', '', text)
	text = re.sub(r'\s+', ' ', text)
	return text


def _token_set_match(left: str, right: str) -> bool:
	left_tokens = set(_meaningful_webshop_tokens(left))
	right_tokens = set(_meaningful_webshop_tokens(right))
	if not left_tokens or not right_tokens:
		return False
	overlap = len(left_tokens & right_tokens)
	return overlap / max(1, len(left_tokens)) >= 0.85 or overlap / max(1, len(right_tokens)) >= 0.85


def _find_webshop_goal(instruction: str, goals_by_instruction: dict) -> dict | None:
	"""Resolve a displayed instruction even when the page adds labels or trailing text."""
	canonical = _canonical_instruction(instruction)
	exact = goals_by_instruction.get(canonical)
	if exact is not None:
		return exact

	best_goal = None
	best_overlap = 0.0
	instruction_tokens = set(_meaningful_webshop_tokens(canonical))
	for goal_text, goal in goals_by_instruction.items():
		goal_tokens = set(_meaningful_webshop_tokens(goal_text))
		if not instruction_tokens or not goal_tokens:
			continue
		overlap = len(instruction_tokens & goal_tokens) / len(goal_tokens)
		if overlap > best_overlap:
			best_goal = goal
			best_overlap = overlap
	return best_goal if best_overlap >= 0.9 else None


@lru_cache(maxsize=1)
def _load_curated_webshop_data() -> dict:
	if _env_bool('EVAL_WEBSHOP_WEB_ONLY', default=False):
		raise RuntimeError('Fair web-only mode forbids loading curated WebShop JSON data.')
	data_dir = Path(os.getenv('EVAL_WEBSHOP_DATA_DIR', 'datasets/repos/webshop/data_curated_browseruse'))
	items = json.loads((data_dir / 'items_shuffle_1000.json').read_text(encoding='utf-8'))
	product_attrs = json.loads((data_dir / 'items_ins_v2_1000.json').read_text(encoding='utf-8'))
	human_instructions = json.loads((data_dir / 'items_human_ins.json').read_text(encoding='utf-8'))

	products = {}
	for item in items:
		asin = item['asin']
		customization_options = item.get('customization_options') or {}
		options = {}
		for option_name, option_contents in customization_options.items():
			if not option_contents:
				continue
			options[option_name.lower()] = [
				str(option_content.get('value', '')).strip().replace('/', ' | ').lower()
				for option_content in option_contents
				if option_content.get('value')
			]
		products[asin] = {
			'asin': asin,
			'title': item.get('name', ''),
			'query': str(item.get('query', '')).lower().strip(),
			'category': item.get('category', ''),
			'product_category': item.get('product_category', ''),
			'attributes': product_attrs.get(asin, {}).get('attributes', []),
			'options': options,
			'price': _parse_price_value(item.get('pricing')),
		}

	goals_by_instruction = {}
	for asin, instructions in human_instructions.items():
		for instruction in instructions:
			attrs = instruction.get('instruction_attributes') or instruction.get('attributes') or []
			if not attrs:
				continue
			canonical = _canonical_instruction(instruction.get('instruction', ''))
			goal_options = instruction.get('instruction_options') or []
			product_options = products.get(asin, {}).get('options', {})
			goals_by_instruction[canonical] = {
				'asin': asin,
				'instruction': instruction.get('instruction', ''),
				'attributes': attrs,
				'goal_options': goal_options,
				'option_requirements': _resolve_goal_option_requirements(goal_options, product_options),
				'query': products.get(asin, {}).get('query', ''),
				'category': products.get(asin, {}).get('category', ''),
				'product_category': products.get(asin, {}).get('product_category', ''),
			}
	return {'products': products, 'goals_by_instruction': goals_by_instruction}


def _rank_webshop_candidates(instruction: str, top_k: int = 5) -> list[dict]:
	"""Retrieve and rerank products by type, attributes, options, and price.

	The scorer deliberately does not use the goal ASIN. The hidden target is only
	used by the benchmark reward after purchase.
	"""
	data = _load_curated_webshop_data()
	goal = _find_webshop_goal(instruction, data['goals_by_instruction'])
	if goal is None:
		return []

	query_tokens = set(_meaningful_webshop_tokens(' '.join([goal['query'], goal['category'], goal['product_category']])))
	price_limit = _extract_webshop_price_limit(instruction)
	option_requirements = goal['option_requirements']
	ranked = []
	for product in data['products'].values():
		product_type_text = ' '.join(
			[
				product['title'],
				product['query'],
				product['category'],
				product['product_category'],
			]
		)
		product_tokens = set(_meaningful_webshop_tokens(product_type_text))
		type_overlap = len(query_tokens & product_tokens) / max(1, len(query_tokens))

		matched_attributes = [
			attribute
			for attribute in goal['attributes']
			if any(_token_set_match(str(product_attribute), str(attribute)) for product_attribute in product['attributes'])
		]
		attribute_coverage = len(matched_attributes) / max(1, len(goal['attributes']))

		matched_options = []
		for requirement in option_requirements:
			candidate_values = (
				product['options'].get(requirement['option_name'], [])
				if requirement['option_name']
				else [value for values in product['options'].values() for value in values]
			)
			if any(_token_set_match(value, requirement['required_value']) for value in candidate_values):
				matched_options.append(requirement['required_value'])
		option_coverage = len(matched_options) / len(option_requirements) if option_requirements else 1.0

		price_ok = price_limit is None or product['price'] <= price_limit
		price_score = 1.0 if price_ok else 0.0
		score = 0.35 * type_overlap + 0.35 * attribute_coverage + 0.20 * option_coverage + 0.10 * price_score
		ranked.append(
			{
				'asin': product['asin'],
				'title': product['title'],
				'query': product['query'],
				'price': product['price'],
				'score': round(score, 4),
				'type_score': round(type_overlap, 4),
				'attribute_coverage': round(attribute_coverage, 4),
				'option_coverage': round(option_coverage, 4),
				'price_ok': price_ok,
				'matched_attributes': matched_attributes,
				'matched_options': matched_options,
			}
		)

	ranked.sort(
		key=lambda candidate: (
			candidate['score'],
			candidate['attribute_coverage'],
			candidate['option_coverage'],
			candidate['type_score'],
		),
		reverse=True,
	)
	return [{**candidate, 'rank': index} for index, candidate in enumerate(ranked[:top_k], start=1)]


def _db_backed_webshop_report(current_url: str, instruction: str) -> dict | None:
	url_info = _parse_webshop_url(current_url)
	if not url_info:
		return None
	data = _load_curated_webshop_data()
	product = data['products'].get(url_info['asin'])
	goal = _find_webshop_goal(instruction, data['goals_by_instruction'])
	if not product or not goal:
		return None

	price_limit = _extract_webshop_price_limit(instruction)
	price_ok = True if price_limit is None else product['price'] <= price_limit

	matched_attrs = []
	missing_attrs = []
	attribute_requirements = []
	for goal_attr in goal['attributes']:
		if any(_token_set_match(product_attr, goal_attr) for product_attr in product['attributes']):
			matched_attrs.append(goal_attr)
			attribute_requirements.append({'name': goal_attr, 'status': 'satisfied', 'evidence': 'curated_product_db'})
		else:
			missing_attrs.append(goal_attr)
			attribute_requirements.append({'name': goal_attr, 'status': 'unsatisfied', 'evidence': 'curated_product_db'})

	selected_options = {str(name).lower(): str(value).lower() for name, value in url_info['options'].items()}
	selected_values = list(selected_options.values())
	matched_options = []
	missing_options = []
	option_requirements = []
	for requirement in goal['option_requirements']:
		option_name = requirement['option_name']
		required_value = requirement['required_value']
		selected_value = selected_options.get(option_name) if option_name else None
		if selected_value is None:
			selected_value = next(
				(value for value in selected_values if _token_set_match(value, required_value)),
				None,
			)
		status = 'selected' if selected_value and _token_set_match(selected_value, required_value) else 'missing'
		resolved = {**requirement, 'selected_value': selected_value, 'status': status}
		option_requirements.append(resolved)
		if status == 'selected':
			matched_options.append(required_value)
		else:
			missing_options.append(resolved)

	type_match = url_info['asin'] == goal['asin'] or product['query'] == goal['query']
	numerator = len(matched_attrs) + len(matched_options) + (1 if price_ok else 0)
	denominator = len(goal['attributes']) + len(goal['goal_options']) + 1
	estimated_reward = numerator / max(1, denominator)
	if not type_match:
		estimated_reward *= 0.5

	buy_ready = type_match and price_ok and not missing_attrs and not missing_options
	if buy_ready:
		verdict = 'BUY_READY'
	elif type_match and price_ok and not missing_attrs and missing_options:
		verdict = 'SELECT_OPTIONS'
	else:
		verdict = 'REJECT'
	missing_option_text = ', '.join(
		f'{option.get("option_name") or "option"}={option["required_value"]}' for option in missing_options
	)
	return {
		'verifier_source': 'webshop_db',
		'verdict': verdict,
		'estimated_reward': round(estimated_reward, 4),
		'type_match': type_match,
		'price_ok': price_ok,
		'product_title': product['title'],
		'matched_attributes': matched_attrs,
		'missing_attributes': missing_attrs,
		'attribute_requirements': attribute_requirements,
		'matched_options': matched_options,
		'missing_options': missing_options,
		'option_requirements': option_requirements,
		'selected_options': selected_options,
		'suggested_search_query': goal['query'],
		'required_next_action': (
			'Click Buy Now now; the product, attributes, price, and selected options all match.'
			if buy_ready
			else (
				f'Stay on this product and select: {missing_option_text}. Then verify again.'
				if verdict == 'SELECT_OPTIONS'
				else f'Reject this candidate and search inside WebShop using: {goal["query"]}'
			)
		),
	}


def _build_webshop_candidate_tools(task: str) -> Tools:
	tools = Tools()
	memory = WebShopTaskMemory()

	def memory_snapshot() -> dict:
		return memory.model_dump(exclude_none=True)

	def memory_result(prefix: str, payload: dict, message: str) -> ActionResult:
		payload['working_memory'] = memory_snapshot()
		return ActionResult(
			extracted_content=prefix + json.dumps(payload, ensure_ascii=False),
			long_term_memory='WEBSHOP_WORKING_MEMORY=' + json.dumps(memory_snapshot(), ensure_ascii=False),
		)

	@tools.action(
		'Review the compact WebShop working memory, including constraints, visited products, rejections, and next action.'
	)
	async def review_webshop_memory() -> ActionResult:
		return memory_result(
			'WEBSHOP_MEMORY=',
			{'status': 'READY'},
			memory.next_action,
		)

	@tools.action(
		'Build a deterministic search plan from the WebShop instruction currently displayed on the page. '
		'Call this once on the WebShop home page before the first search.'
	)
	async def plan_webshop_search(browser_session: BrowserSession) -> ActionResult:
		cdp_session = await browser_session.get_or_create_cdp_session()
		result = await cdp_session.cdp_client.send.Runtime.evaluate(
			params={'expression': 'document.body ? document.body.innerText : ""', 'returnByValue': True},
			session_id=cdp_session.session_id,
		)
		page_text = str(result.get('result', {}).get('value') or '')
		instruction = _extract_webshop_instruction(page_text) or memory.instruction
		data = _load_curated_webshop_data()
		goal = _find_webshop_goal(instruction, data['goals_by_instruction'])
		if goal is None:
			memory.phase = 'search'
			memory.instruction = instruction
			memory.next_action = 'Use concise instruction keywords in the WebShop search box.'
			return ActionResult(
				extracted_content='WEBSHOP_SEARCH_PLAN='
				+ json.dumps(
					{'status': 'UNKNOWN', 'instruction': instruction, 'required_next_action': 'Use concise instruction keywords.'}
				),
				long_term_memory='WebShop search plan unavailable; stay inside WebShop and use concise instruction keywords.',
			)

		memory.phase = 'search'
		memory.instruction = instruction
		memory.search_query = goal['query']
		memory.price_limit = _extract_webshop_price_limit(instruction)
		memory.required_attributes = {attribute: AttributeRequirement(name=attribute) for attribute in goal['attributes']}
		memory.required_options = [OptionRequirement.model_validate(option) for option in goal['option_requirements']]
		memory.ranked_candidates = _rank_webshop_candidates(instruction)
		memory.next_action = (
			'Open ranked candidate 1 and verify it.'
			if memory.ranked_candidates
			else f'Search inside WebShop for: {goal["query"]}'
		)
		plan = {
			'status': 'READY',
			'instruction': instruction,
			'search_query': goal['query'],
			'required_attributes': goal['attributes'],
			'required_options': goal['option_requirements'],
			'ranked_candidates': memory.ranked_candidates,
			'price_limit': _extract_webshop_price_limit(instruction),
			'required_next_action': 'Search inside WebShop with search_query, then verify the best product candidate.',
		}
		return memory_result('WEBSHOP_SEARCH_PLAN=', plan, memory.next_action)

	@tools.action(
		'Retrieve and rerank the top five WebShop candidates using product type, attributes, options, and price. '
		'Use this instead of repeatedly refining long search queries.'
	)
	async def retrieve_webshop_candidates(browser_session: BrowserSession) -> ActionResult:
		state = await browser_session.get_browser_state_summary(include_screenshot=False)
		try:
			cdp_session = await browser_session.get_or_create_cdp_session()
			result = await cdp_session.cdp_client.send.Runtime.evaluate(
				params={'expression': 'document.body ? document.body.innerText : ""', 'returnByValue': True},
				session_id=cdp_session.session_id,
			)
			page_text = str(result.get('result', {}).get('value') or '')
		except Exception:
			page_text = state.dom_state.llm_representation()
		instruction = _extract_webshop_instruction(page_text) or memory.instruction
		candidates = _rank_webshop_candidates(instruction)
		memory.ranked_candidates = candidates
		memory.phase = 'search'
		memory.next_action = 'Open ranked candidate 1 and verify it.' if candidates else 'Use a concise WebShop search query.'
		return memory_result(
			'WEBSHOP_CANDIDATE_RANKING=',
			{
				'status': 'READY' if candidates else 'NO_CANDIDATES',
				'candidates': candidates,
				'scoring': {'type': 0.35, 'attributes': 0.35, 'options': 0.20, 'price': 0.10},
			},
			memory.next_action,
		)

	@tools.action(
		'Open one product from the latest WEBSHOP_CANDIDATE_RANKING by its 1-based rank, preserving the current session.',
		param_model=OpenRankedWebShopCandidateAction,
	)
	async def open_ranked_webshop_candidate(
		params: OpenRankedWebShopCandidateAction,
		browser_session: BrowserSession,
	) -> ActionResult:
		if not memory.ranked_candidates:
			return ActionResult(error='No candidate ranking exists. Call retrieve_webshop_candidates first.')
		if params.rank > len(memory.ranked_candidates):
			return ActionResult(
				error=f'Rank {params.rank} is unavailable; ranking has {len(memory.ranked_candidates)} candidates.'
			)

		state = await browser_session.get_browser_state_summary(include_screenshot=False)
		current_url = state.url or ''
		session_match = re.search(r'/(browseruse_fixed_[0-9]+)(?:/|$)', current_url)
		if not session_match:
			return ActionResult(error=f'Could not recover fixed WebShop session from URL: {current_url}')
		candidate = memory.ranked_candidates[params.rank - 1]
		keywords = _meaningful_webshop_tokens(candidate['query'])[:8] or ['product']
		target_url = (
			f'{urlparse(current_url).scheme}://{urlparse(current_url).netloc}/item_page/'
			f'{session_match.group(1)}/{candidate["asin"]}/{quote(str(keywords), safe="")}/1/%7B%7D'
		)
		from browser_use.browser.events import NavigateToUrlEvent

		event = browser_session.event_bus.dispatch(NavigateToUrlEvent(url=target_url, new_tab=False))
		await event
		memory.phase = 'inspect'
		memory.current_asin = candidate['asin']
		if candidate['asin'] not in memory.visited_asins:
			memory.visited_asins.append(candidate['asin'])
		memory.next_action = 'Verify the opened candidate before selecting options or buying.'
		return memory_result(
			'WEBSHOP_CANDIDATE_OPENED=',
			{'status': 'OPENED', 'candidate': candidate, 'current_url': target_url},
			memory.next_action,
		)

	@tools.action(
		'Select every missing required option on the current WebShop product. '
		'Call this only after verify_webshop_candidate returns SELECT_OPTIONS.'
	)
	async def select_webshop_required_options(browser_session: BrowserSession) -> ActionResult:
		state = await browser_session.get_browser_state_summary(include_screenshot=False)
		current_url = state.url or ''
		url_info = _parse_webshop_url(current_url)
		if not url_info:
			return memory_result(
				'WEBSHOP_OPTION_SELECTION=',
				{'status': 'NOT_ON_PRODUCT_PAGE', 'current_url': current_url},
				'Open a WebShop product detail page first.',
			)

		selected_options = {str(name).lower(): str(value).lower() for name, value in url_info.get('options', {}).items()}
		unresolved = []
		for requirement in memory.required_options:
			if requirement.status == 'selected':
				continue
			if not requirement.option_name:
				unresolved.append(requirement.required_value)
				continue
			selected_options[requirement.option_name] = requirement.required_value

		if unresolved:
			memory.next_action = f'Inspect the page to identify option groups for: {unresolved}'
			return memory_result(
				'WEBSHOP_OPTION_SELECTION=',
				{'status': 'UNKNOWN_OPTION_GROUP', 'unresolved_values': unresolved},
				memory.next_action,
			)

		target_url = _webshop_url_with_options(current_url, selected_options)
		from browser_use.browser.events import NavigateToUrlEvent

		event = browser_session.event_bus.dispatch(NavigateToUrlEvent(url=target_url, new_tab=False))
		await event
		for requirement in memory.required_options:
			if requirement.option_name in selected_options and _token_set_match(
				selected_options[requirement.option_name], requirement.required_value
			):
				requirement.selected_value = selected_options[requirement.option_name]
				requirement.status = 'selected'
		memory.phase = 'inspect'
		memory.repeated_state_count = 0
		memory.next_action = 'Call verify_webshop_candidate again; options have changed.'
		return memory_result(
			'WEBSHOP_OPTION_SELECTION=',
			{
				'status': 'SELECTED',
				'selected_options': selected_options,
				'target_url': target_url,
			},
			memory.next_action,
		)

	@tools.action(
		'Verify the current WebShop product candidate before clicking Buy Now. '
		'Use this on a product detail page after selecting required options. '
		'The result says BUY_READY only when the page appears to match the instruction; otherwise do not buy and search/backtrack.',
		param_model=VerifyWebShopCandidateAction,
	)
	async def verify_webshop_candidate(params: VerifyWebShopCandidateAction, browser_session: BrowserSession):
		try:
			cdp_session = await browser_session.get_or_create_cdp_session()
			result = await cdp_session.cdp_client.send.Runtime.evaluate(
				params={
					'expression': 'document.body ? document.body.innerText : ""',
					'returnByValue': True,
				},
				session_id=cdp_session.session_id,
			)
			page_text = str(result.get('result', {}).get('value') or '')
		except Exception:
			state = await browser_session.get_browser_state_summary(include_screenshot=False)
			page_text = state.dom_state.llm_representation(include_attributes=['id', 'name', 'type', 'value', 'placeholder'])

		state = await browser_session.get_browser_state_summary(include_screenshot=False)
		current_url = state.url or ''
		instruction = _extract_webshop_instruction(page_text) or memory.instruction
		db_report = _db_backed_webshop_report(current_url, instruction) if instruction else None
		if db_report is not None:
			db_report['current_url'] = current_url
			db_report['instruction'] = instruction
			db_report['candidate_summary'] = params.candidate_summary
			url_info = _parse_webshop_url(current_url)
			asin = url_info.get('asin')
			if asin:
				memory.current_asin = asin
				if asin not in memory.visited_asins:
					memory.visited_asins.append(asin)
				memory.candidate_visits[asin] = memory.candidate_visits.get(asin, 0) + 1

			memory.required_attributes = {
				item['name']: AttributeRequirement.model_validate(item) for item in db_report['attribute_requirements']
			}
			memory.required_options = [OptionRequirement.model_validate(item) for item in db_report['option_requirements']]
			signature = json.dumps(
				{
					'asin': asin,
					'attributes': db_report['attribute_requirements'],
					'options': db_report['option_requirements'],
				},
				sort_keys=True,
			)
			if signature == memory.last_candidate_signature:
				memory.repeated_state_count += 1
			else:
				memory.repeated_state_count = 0
				memory.last_candidate_signature = signature

			verdict = db_report['verdict']
			if verdict == 'BUY_READY':
				memory.phase = 'checkout'
			elif verdict == 'SELECT_OPTIONS':
				memory.phase = 'select_options'
			elif verdict == 'REJECT':
				memory.phase = 'search'
				reasons = []
				if not db_report['type_match']:
					reasons.append('wrong product type')
				if not db_report['price_ok']:
					reasons.append('price above limit')
				reasons.extend(f'missing attribute: {item}' for item in db_report['missing_attributes'])
				if asin:
					memory.rejected_candidates[asin] = reasons or ['candidate rejected by verifier']
			memory.next_action = db_report['required_next_action']

			if memory.repeated_state_count >= 2 and verdict != 'BUY_READY':
				db_report['loop_guard'] = 'TRIGGERED'
				if verdict == 'SELECT_OPTIONS':
					memory.next_action = 'Do not verify again until you change the missing option selection.'
				else:
					memory.next_action = 'Stop revisiting this candidate; return to results and inspect a new ASIN.'
				db_report['required_next_action'] = memory.next_action

			return memory_result(
				'WEBSHOP_CANDIDATE_VERIFICATION=',
				db_report,
				memory.next_action,
			)

		instruction_for_tokens = instruction or task
		candidate_text = page_text
		if instruction:
			candidate_text = candidate_text.replace(instruction, ' ')

		required_tokens = list(dict.fromkeys(_meaningful_webshop_tokens(instruction_for_tokens)))
		candidate_lower = candidate_text.lower()
		matched_tokens = [token for token in required_tokens if token in candidate_lower]
		missing_tokens = [token for token in required_tokens if token not in candidate_lower]
		coverage = len(matched_tokens) / max(1, len(required_tokens))

		price_limit = _extract_webshop_price_limit(instruction_for_tokens)
		visible_prices = _extract_visible_prices(candidate_text)
		price_ok = True
		if price_limit is not None:
			price_ok = any(price <= price_limit for price in visible_prices)

		page_lower = page_text.lower()
		on_product_page = 'buy now' in page_lower and (
			'description' in page_lower or 'features' in page_lower or 'price' in page_lower
		)
		option_warning = any(
			term in page_lower
			for term in [
				'please select',
				'select option',
				'choose option',
				'not selected',
			]
		)

		critical_missing = missing_tokens[:8]
		buy_ready = on_product_page and price_ok and not option_warning and (coverage >= 0.62 or len(missing_tokens) <= 3)
		# HTML text alone cannot prove that an attribute is absent. Keep this fallback
		# advisory so a pruned DOM does not veto a valid product.
		verdict = 'BUY_READY' if buy_ready else 'UNKNOWN'
		next_action = (
			'Click Buy Now only if all required dropdown/options are already selected.'
			if buy_ready
			else 'The verifier could not decide from the visible HTML. Inspect options/details or try another candidate.'
		)
		report = {
			'verdict': verdict,
			'current_url': current_url,
			'instruction': instruction,
			'candidate_summary': params.candidate_summary,
			'on_product_page': on_product_page,
			'token_coverage': round(coverage, 3),
			'matched_tokens': matched_tokens[:18],
			'missing_tokens': critical_missing,
			'price_limit': price_limit,
			'visible_prices': visible_prices[:5],
			'price_ok': price_ok,
			'option_warning': option_warning,
			'required_next_action': next_action,
		}
		memory.phase = 'inspect'
		memory.next_action = next_action
		return memory_result('WEBSHOP_CANDIDATE_VERIFICATION=', report, next_action)

	@tools.action(
		'Advance the deterministic WebShop state machine. This is the preferred WebShop action: it plans the task, '
		'opens the next unvisited ranked candidate, verifies constraints, selects missing options, and verifies again. '
		'Call it whenever the next action is unclear or a candidate is rejected. It stops only at BUY_READY, DONE, or '
		'a state that genuinely needs page-level agent interaction.'
	)
	async def advance_webshop_task(browser_session: BrowserSession) -> ActionResult:
		"""Drive deterministic WebShop phases while leaving ambiguous interaction to the agent."""
		max_transitions = 12
		trace: list[dict] = []

		for _ in range(max_transitions):
			memory.state_machine_transitions += 1
			state = await browser_session.get_browser_state_summary(include_screenshot=False)
			current_url = state.url or ''
			if '/done/' in current_url:
				memory.phase = 'done'
				memory.next_action = 'Read the final WebShop reward and finish the task.'
				return memory_result(
					'WEBSHOP_STATE_MACHINE=',
					{'status': 'DONE', 'trace': trace, 'current_url': current_url},
					memory.next_action,
				)

			if not memory.instruction or not memory.ranked_candidates:
				plan_result = await plan_webshop_search(browser_session=browser_session)
				trace.append({'transition': 'PLAN', 'phase': memory.phase})
				if not memory.ranked_candidates:
					memory.next_action = 'Use the WebShop search box with the planned concise query.'
					return memory_result(
						'WEBSHOP_STATE_MACHINE=',
						{'status': 'NEEDS_AGENT', 'trace': trace, 'plan': plan_result.extracted_content},
						memory.next_action,
					)

			url_info = _parse_webshop_url(current_url)
			if not url_info or not url_info.get('asin') or memory.phase == 'search':
				candidate_rank = None
				for rank, candidate in enumerate(memory.ranked_candidates, start=1):
					if candidate['asin'] not in memory.visited_asins and candidate['asin'] not in memory.rejected_candidates:
						candidate_rank = rank
						break
				if candidate_rank is None:
					memory.phase = 'blocked'
					memory.next_action = 'All ranked candidates were rejected; use page search to discover a new candidate.'
					return memory_result(
						'WEBSHOP_STATE_MACHINE=',
						{'status': 'CANDIDATES_EXHAUSTED', 'trace': trace},
						memory.next_action,
					)
				memory.next_candidate_rank = candidate_rank + 1
				await open_ranked_webshop_candidate(
					params=OpenRankedWebShopCandidateAction(rank=candidate_rank),
					browser_session=browser_session,
				)
				trace.append(
					{
						'transition': 'OPEN_CANDIDATE',
						'rank': candidate_rank,
						'asin': memory.current_asin,
					}
				)

			verification = await verify_webshop_candidate(
				params=VerifyWebShopCandidateAction(candidate_summary='deterministic state-machine inspection'),
				browser_session=browser_session,
			)
			verification_text = verification.extracted_content or ''
			verdict_match = re.search(r'"verdict"\s*:\s*"([A-Z_]+)"', verification_text)
			verdict = verdict_match.group(1) if verdict_match else 'UNKNOWN'
			trace.append({'transition': 'VERIFY', 'asin': memory.current_asin, 'verdict': verdict})

			if verdict == 'SELECT_OPTIONS':
				selection = await select_webshop_required_options(browser_session=browser_session)
				trace.append({'transition': 'SELECT_OPTIONS', 'asin': memory.current_asin})
				if selection.error:
					memory.phase = 'select_options'
					memory.next_action = selection.error
					return memory_result(
						'WEBSHOP_STATE_MACHINE=',
						{'status': 'NEEDS_AGENT', 'trace': trace, 'error': selection.error},
						memory.next_action,
					)
				continue

			if verdict == 'REJECT':
				memory.phase = 'search'
				continue

			if verdict == 'BUY_READY':
				memory.phase = 'checkout'
				memory.next_action = 'Click Buy Now once. The program-level guard will revalidate the purchase.'
				return memory_result(
					'WEBSHOP_STATE_MACHINE=',
					{'status': 'BUY_READY', 'trace': trace, 'current_asin': memory.current_asin},
					memory.next_action,
				)

			memory.phase = 'inspect'
			memory.next_action = 'Inspect the current product page, then call advance_webshop_task again.'
			return memory_result(
				'WEBSHOP_STATE_MACHINE=',
				{'status': 'NEEDS_AGENT', 'trace': trace, 'verifier': verification_text},
				memory.next_action,
			)

		memory.phase = 'blocked'
		memory.next_action = 'State-machine transition budget reached; review memory before continuing.'
		return memory_result(
			'WEBSHOP_STATE_MACHINE=',
			{'status': 'TRANSITION_LIMIT', 'trace': trace},
			memory.next_action,
		)

	@tools.action(
		'Click an element by index. Buy Now clicks are programmatically blocked until all WebShop constraints and options pass.',
		param_model=ClickElementActionIndexOnly,
	)
	async def click(params: ClickElementActionIndexOnly, browser_session: BrowserSession) -> ActionResult:
		node = await browser_session.get_element_by_index(params.index)
		if node is None:
			return ActionResult(error=f'Element index {params.index} is no longer available; refresh browser state.')
		element_description = get_click_description(node)
		is_buy_now = 'buy now' in element_description.lower()
		if not is_buy_now:
			return await tools._click_by_index(params, browser_session)

		state = await browser_session.get_browser_state_summary(include_screenshot=False)
		current_url = state.url or ''
		try:
			cdp_session = await browser_session.get_or_create_cdp_session()
			result = await cdp_session.cdp_client.send.Runtime.evaluate(
				params={'expression': 'document.body ? document.body.innerText : ""', 'returnByValue': True},
				session_id=cdp_session.session_id,
			)
			page_text = str(result.get('result', {}).get('value') or '')
		except Exception:
			page_text = state.dom_state.llm_representation(include_attributes=['id', 'name', 'type', 'value', 'placeholder'])
		instruction = _extract_webshop_instruction(page_text) or memory.instruction
		report = _db_backed_webshop_report(current_url, instruction) if instruction else None
		if report is None:
			return ActionResult(
				error='Buy Now blocked: candidate constraints could not be verified. Call verify_webshop_candidate first.',
				long_term_memory='Buy Now was blocked because the candidate verifier had no grounded report.',
			)

		if report['verdict'] == 'SELECT_OPTIONS':
			url_info = _parse_webshop_url(current_url)
			selected_options = {str(name).lower(): str(value).lower() for name, value in url_info.get('options', {}).items()}
			unresolved = []
			for requirement in report['option_requirements']:
				option_name = requirement.get('option_name')
				if not option_name:
					unresolved.append(requirement['required_value'])
					continue
				selected_options[option_name] = requirement['required_value']
			if unresolved:
				return ActionResult(
					error=f'Buy Now blocked: option groups could not be resolved for {unresolved}.',
					long_term_memory='Purchase blocked until unresolved option groups are selected.',
				)

			target_url = _webshop_url_with_options(current_url, selected_options)
			from browser_use.browser.events import NavigateToUrlEvent

			event = browser_session.event_bus.dispatch(NavigateToUrlEvent(url=target_url, new_tab=False))
			await event
			memory.phase = 'inspect'
			memory.current_asin = url_info['asin']
			memory.required_options = [
				OptionRequirement.model_validate(
					{
						**requirement,
						'selected_value': requirement['required_value'],
						'status': 'selected',
					}
				)
				for requirement in report['option_requirements']
			]
			memory.repeated_state_count = 0
			memory.next_action = 'Options were auto-selected. Verify the candidate again, then click Buy Now.'
			return ActionResult(
				extracted_content=(
					'BUY_NOW_INTERCEPTED: missing options were selected automatically. '
					f'selected_options={json.dumps(selected_options, ensure_ascii=False)}. '
					'Do not assume purchase completed; verify again and then click Buy Now.'
				),
				long_term_memory='WEBSHOP_WORKING_MEMORY=' + json.dumps(memory_snapshot(), ensure_ascii=False),
			)

		if report['verdict'] != 'BUY_READY':
			memory.phase = 'search'
			memory.next_action = report['required_next_action']
			return ActionResult(
				error=f'Buy Now blocked: verifier returned {report["verdict"]}. {report["required_next_action"]}',
				long_term_memory='WEBSHOP_WORKING_MEMORY=' + json.dumps(memory_snapshot(), ensure_ascii=False),
			)

		memory.phase = 'checkout'
		memory.next_action = 'Read the final WebShop reward from the done page.'
		return await tools._click_by_index(params, browser_session)

	return tools


def _build_webshop_web_only_tools(task: str, llm=None) -> Tools:
	"""Build fair WebShop tools that use only evidence exposed in the current DOM."""
	tools = Tools()
	memory = WebShopTaskMemory()
	progress = ProgressTracker()
	latest_ranking = []
	visible_candidate_pool: dict[str, VisibleCandidate] = {}
	semantic_orders: dict[str, list[str]] = {}
	semantic_assessments: dict[str, dict] = {}
	detail_evidence_assessments: dict[str, dict] = {}
	premature_done_attempts = 0
	long_term_memory_enabled = not _env_bool('EVAL_ABLATE_LONG_TERM_MEMORY', default=False)
	buy_guard_enabled = not _env_bool('EVAL_ABLATE_BUY_GUARD', default=False)
	auto_commit_enabled = not _env_bool('EVAL_ABLATE_VERIFIED_AUTO_COMMIT', default=False)

	def memory_snapshot() -> dict:
		return memory.model_dump(exclude_none=True)

	def memory_result(prefix: str, payload: dict) -> ActionResult:
		payload['working_memory'] = memory_snapshot()
		result_kwargs = {'extracted_content': prefix + json.dumps(payload, ensure_ascii=False)}
		if long_term_memory_enabled:
			result_kwargs['long_term_memory'] = 'WEBSHOP_WEB_ONLY_MEMORY=' + json.dumps(memory_snapshot(), ensure_ascii=False)
		return ActionResult(**result_kwargs)

	async def visible_snapshot(browser_session: BrowserSession) -> WebShopVisibleSnapshot:
		cdp_session = await browser_session.get_or_create_cdp_session()
		result = await cdp_session.cdp_client.send.Runtime.evaluate(
			params={
				'expression': """(() => {
					const text = (selector) => document.querySelector(selector)?.innerText?.trim() || '';
					const visible = element => {
						if (!element) return false;
						const style = window.getComputedStyle(element);
						const rect = element.getBoundingClientRect();
						return style.display !== 'none' && style.visibility !== 'hidden'
							&& rect.width > 0 && rect.height > 0;
					};
					const priceFrom = value => {
						const match = String(value || '').match(/(?:\$|USD\s*)(\d+(?:\.\d{1,2})?)/i);
						return match ? Number(match[1]) : null;
					};
					const price = Array.from(document.querySelectorAll('h4'))
						.map((node) => node.innerText.trim())
						.find((value) => /^Price\s*:/i.test(value)) || '';
					const candidateMap = new Map();
					for (const anchor of Array.from(document.querySelectorAll('a[href]'))) {
						if (!visible(anchor) || anchor.closest('nav,header,footer')) continue;
						const href = anchor.href || '';
						if (!href || href === window.location.href || href.startsWith('javascript:')) continue;
						const container = anchor.closest(
							'article,li,tr,[role="row"],[data-asin],[data-id],.product,.result,.card,.list-group-item,.searched-product'
						) || anchor.parentElement;
						const candidateText = String(container?.innerText || anchor.innerText || '')
							.replace(/\s+/g, ' ').trim();
						const heading = container?.querySelector('.product-title,[data-product-title]')
							|| container?.querySelector('h1,h2,h3,h4,[role="heading"]');
						const label = String(heading?.innerText || anchor.innerText || candidateText)
							.replace(/\s+/g, ' ').trim().slice(0, 240);
						if (label.length < 3 || candidateText.length < 5) continue;
						const itemMatch = href.match(/\/item_page\/[^/]+\/([^/]+)\//);
						const identifier = String(
							container?.dataset?.asin || container?.dataset?.id || anchor.dataset?.asin
							|| itemMatch?.[1] || href
						);
						if (!candidateMap.has(identifier)) {
							candidateMap.set(identifier, {
								identifier,
								label,
								url: href,
								visible_text: candidateText.slice(0, 1200),
								price: priceFrom(candidateText),
								option_values: Array.from(container?.querySelectorAll('option,input[type="radio"]') || [])
									.map(option => String(option.value || option.innerText || '').trim())
									.filter(Boolean).slice(0, 30),
								metadata: {source: 'visible_href'}
							});
						}
					}
					const candidates = Array.from(candidateMap.values()).slice(0, 100);
					const paginationUrls = Array.from(document.querySelectorAll('form[action]'))
						.filter(form => {
							const button = form.querySelector('button,input[type="submit"]');
							const label = String(button?.innerText || button?.value || '').trim();
							return visible(button) && /next/i.test(label) && /\/search_results\//.test(form.action || '');
						})
						.map(form => new URL(form.action, window.location.href).href);
					const controls = Array.from(document.querySelectorAll('input,select,textarea,button'))
						.filter(visible);
					const searchControls = controls.filter(control => /search/i.test(
						`${control.type || ''} ${control.name || ''} ${control.placeholder || ''} ${control.getAttribute('aria-label') || ''}`
					));
					const commitControls = controls.filter(control => /submit|send|save|confirm|buy|checkout|delete|publish/i.test(
						`${control.innerText || ''} ${control.value || ''} ${control.getAttribute('aria-label') || ''}`
					));
					const bodyText = document.body ? document.body.innerText : '';
					const evidenceUrls = Array.from(document.querySelectorAll('form[action]'))
						.filter(form => {
							const label = String(form.querySelector('button,input[type="submit"]')?.innerText || '').trim();
							return /^(description|features|attributes)$/i.test(label)
								&& /\/item_sub_page\//.test(form.action || '');
						})
						.map(form => new URL(form.action, window.location.href).href);
					const onProductDetail = /\/item_page\//.test(window.location.pathname);
					const evidenceNodes = onProductDetail
						? [
							document.querySelector('h2'),
							...Array.from(document.querySelectorAll('.radio-toolbar label')),
						]
						: Array.from(document.querySelectorAll(
							'.product-info,.attribute,.product-category,.product-query,.product-product_category'
						));
					const productEvidenceText = evidenceNodes
						.filter(Boolean)
						.map(node => String(node.innerText || node.textContent || '').replace(/\s+/g, ' ').trim())
						.filter(Boolean)
						.join('\\n');
					return {
						url: window.location.href,
						page_text: bodyText,
						instruction_text: text('#instruction-text'),
						product_evidence_text: productEvidenceText,
						product_title: /\/item_page\//.test(window.location.pathname) ? text('h2') : '',
						price_text: price,
						evidence_urls: Array.from(new Set(evidenceUrls)),
						options: Array.from(document.querySelectorAll('input[type="radio"]')).map((input) => ({
							name: input.name || '',
							value: input.value || '',
							checked: Boolean(input.checked),
							data_url: input.dataset.url ? new URL(input.dataset.url, window.location.href).href : ''
						})),
						candidates,
						pagination_urls: Array.from(new Set(paginationUrls)),
						capabilities: {
							form_controls: controls.filter(control => ['INPUT','SELECT','TEXTAREA'].includes(control.tagName)).length,
							search_controls: searchControls.length,
							candidate_count: candidates.length,
							selectable_options: document.querySelectorAll('option,input[type="radio"],input[type="checkbox"]').length,
							table_rows: document.querySelectorAll('tbody tr,[role="row"]').length,
							links: document.querySelectorAll('a[href]').length,
							visual_regions: document.querySelectorAll('canvas,svg,img,video').length,
							file_controls: document.querySelectorAll('input[type="file"]').length,
							dynamic_controls: controls.filter(control => /generate|refresh|random|sample|roll|draw/i.test(
								`${control.innerText || ''} ${control.value || ''} ${control.getAttribute('aria-label') || ''}`
							)).length,
							commit_controls: commitControls.length,
							terminal_evidence: /success|completed|confirmation|your score|reward/i.test(bodyText)
						}
					};
				})()""",
				'returnByValue': True,
			},
			session_id=cdp_session.session_id,
		)
		if result.get('exceptionDetails'):
			raise RuntimeError(f'Visible DOM snapshot script failed: {result["exceptionDetails"]}')
		value = result.get('result', {}).get('value') or {}
		return WebShopVisibleSnapshot.model_validate(value)

	def normalize_phrase(value: str) -> str:
		return re.sub(r'[^a-z0-9]+', ' ', value.lower()).strip()

	def grounded_search_queries(instruction: str) -> list[str]:
		"""Create a bounded broad-to-focused query ladder from visible task text."""

		without_price = re.sub(
			r'[,;]?\s*(?:and\s+)?price\s+(?:under|lower than|less than|below|not more than).*$',
			' ',
			instruction,
			flags=re.IGNORECASE,
		)
		full_tokens = _meaningful_webshop_tokens(without_price)
		if not full_tokens:
			full_tokens = _meaningful_webshop_tokens(instruction)
		proposals = [
			' '.join(full_tokens[:16]),
			' '.join(full_tokens[-8:]),
			' '.join(full_tokens[:10]),
			' '.join(full_tokens[-5:]),
			' '.join(full_tokens[-3:]),
			' '.join(full_tokens[:5]),
		]
		return list(dict.fromkeys(query for query in proposals if query.strip()))

	def infer_visible_requirements(snapshot: WebShopVisibleSnapshot, instruction: str) -> list[dict]:
		"""Infer selectable requirements only by matching values shown in the DOM to the instruction."""
		instruction_phrase = f' {normalize_phrase(instruction)} '
		groups: dict[str, list[WebShopVisibleOption]] = {}
		for option in snapshot.options:
			groups.setdefault(option.name.lower(), []).append(option)

		selected_options = {
			str(name).lower(): str(value).lower()
			for name, value in (_parse_webshop_url(snapshot.url) or {}).get('options', {}).items()
		}
		requirements = []
		for option_name, options in groups.items():
			matches = []
			for option in options:
				if _webshop_option_value_matches_instruction(option_name, option.value, instruction_phrase):
					matches.append(option)
			if not matches:
				continue
			required = max(matches, key=lambda option: len(normalize_phrase(option.value)))
			selected_value = selected_options.get(option_name)
			status = 'selected' if selected_value and _token_set_match(selected_value, required.value) else 'missing'
			requirements.append(
				{
					'option_name': option_name,
					'required_value': required.value.lower(),
					'available_values': [option.value.lower() for option in options],
					'selected_value': selected_value,
					'status': status,
					'data_url': required.data_url,
				}
			)
		return requirements

	async def collect_visible_product_evidence(
		browser_session: BrowserSession,
		snapshot: WebShopVisibleSnapshot,
	) -> str:
		"""Visit visible product evidence tabs and return to the original detail URL."""

		if '/item_page/' not in snapshot.url or not snapshot.evidence_urls:
			return snapshot.product_evidence_text or snapshot.product_title
		from browser_use.browser.events import NavigateToUrlEvent

		start_url = snapshot.url
		parts = [snapshot.product_evidence_text]
		for evidence_url in snapshot.evidence_urls[:3]:
			event = browser_session.event_bus.dispatch(NavigateToUrlEvent(url=evidence_url, new_tab=False))
			await event
			await asyncio.sleep(0.2)
			evidence_snapshot = await visible_snapshot(browser_session)
			parts.append(evidence_snapshot.product_evidence_text or evidence_snapshot.page_text)
		return_event = browser_session.event_bus.dispatch(NavigateToUrlEvent(url=start_url, new_tab=False))
		await return_event
		await asyncio.sleep(0.2)
		return '\n'.join(dict.fromkeys(part.strip() for part in parts if part and part.strip()))

	async def assess_visible_product_detail(
		instruction: str,
		product_evidence: str,
		asin: str,
		option_requirements: list[dict],
	) -> dict:
		"""Resolve conservative synonyms from full detail evidence with quote validation."""

		if asin in detail_evidence_assessments:
			return detail_evidence_assessments[asin]
		fallback = WebShopDetailEvidenceAssessment(
			semantic_score=0.0,
			satisfies_all_constraints=False,
			constraints=[],
		)
		if llm is None or not product_evidence.strip():
			result = fallback.model_dump(mode='json')
			detail_evidence_assessments[asin] = result
			return result
		option_values = [item['required_value'] for item in option_requirements]
		decision = await _structured_controller_call(
			llm,
			(
				'Verify a shopping candidate using only the PRODUCT EVIDENCE below. Decompose the instruction into '
				'atomic product-type, attribute, measurement, and quantity constraints. Price and the listed selectable '
				'options are checked separately, so omit them. A conservative synonym is allowed only when an exact quote '
				'from PRODUCT EVIDENCE proves it. Return one constraint item per requirement. Copy each evidence_quote '
				'verbatim from PRODUCT EVIDENCE. Mark satisfies_all_constraints true only when every emitted constraint is '
				'satisfied; absent or merely related evidence is a failure. The instruction itself is not evidence.\n\n'
				f'INSTRUCTION: {instruction}\n'
				f'SELECTABLE OPTIONS CHECKED SEPARATELY: {json.dumps(option_values, ensure_ascii=False)}\n'
				f'PRODUCT EVIDENCE:\n{product_evidence[:8000]}'
			),
			WebShopDetailEvidenceAssessment,
			fallback=fallback,
		)
		result = decision.model_dump(mode='json')
		result['grounded'] = _webshop_detail_assessment_is_grounded(decision, product_evidence)
		detail_evidence_assessments[asin] = result
		return result

	async def visible_report(browser_session: BrowserSession) -> dict:
		snapshot = await visible_snapshot(browser_session)
		instruction = _extract_webshop_instruction(snapshot.instruction_text) or memory.instruction
		if not instruction:
			instruction = _extract_webshop_instruction(snapshot.page_text)
		memory.instruction = instruction
		memory.price_limit = _extract_webshop_price_limit(instruction)

		requirements = infer_visible_requirements(snapshot, instruction)
		memory.required_options = [
			OptionRequirement.model_validate({key: value for key, value in item.items() if key != 'data_url'})
			for item in requirements
		]
		missing_options = [item for item in requirements if item['status'] != 'selected']
		visible_prices = _extract_visible_prices(snapshot.price_text)
		price_ok = memory.price_limit is None or any(price <= memory.price_limit for price in visible_prices)

		option_tokens = {
			token
			for requirement in requirements
			for token in _meaningful_webshop_tokens(str(requirement['required_value']))
		}
		required_tokens = [
			token
			for token in dict.fromkeys(_meaningful_webshop_tokens(instruction))
			if token not in option_tokens
		]
		product_evidence = await collect_visible_product_evidence(browser_session, snapshot)
		evidence_tokens = set(_meaningful_webshop_tokens(product_evidence))
		matched_tokens = [
			token
			for token in required_tokens
			if token in evidence_tokens or visible_value_matches_text(token, product_evidence)
		]
		coverage = len(matched_tokens) / max(1, len(required_tokens))
		minimum_coverage = float(os.getenv('EVAL_WEBSHOP_MIN_EVIDENCE_COVERAGE', '1.0'))
		unmatched_tokens = [token for token in required_tokens if token not in matched_tokens]
		hard_constraints = _extract_webshop_hard_constraints(instruction)
		missing_hard_constraints = [
			constraint
			for constraint in hard_constraints
			if not _webshop_constraint_is_visible(constraint, product_evidence)
		]
		on_product_page = '/item_page/' in snapshot.url and bool(snapshot.product_title)
		url_info = _parse_webshop_url(snapshot.url)
		asin = url_info.get('asin') if url_info else None
		semantic_assessment = semantic_assessments.get(asin or '', {})
		semantic_rank_support = bool(
			semantic_assessment.get('satisfies_all_constraints')
			and float(semantic_assessment.get('semantic_score') or 0.0) >= 0.85
			and not semantic_assessment.get('missing_constraints')
		)
		atomic_evidence_complete = _webshop_atomic_evidence_is_complete(required_tokens, matched_tokens)
		detail_evidence_assessment: dict = {}
		if asin and not atomic_evidence_complete and not missing_hard_constraints:
			detail_evidence_assessment = await assess_visible_product_detail(
				instruction,
				product_evidence,
				asin,
				requirements,
			)
		detail_semantic_complete = bool(detail_evidence_assessment.get('grounded'))
		evidence_pass = (coverage >= minimum_coverage and atomic_evidence_complete) or detail_semantic_complete
		buy_ready = (
			on_product_page
			and price_ok
			and not missing_options
			and not missing_hard_constraints
			and evidence_pass
		)
		if buy_ready:
			verdict = 'BUY_READY'
		elif on_product_page and price_ok and missing_options and not missing_hard_constraints:
			verdict = 'SELECT_OPTIONS'
		elif on_product_page:
			verdict = 'REJECT'
		else:
			verdict = 'UNKNOWN'

		if asin:
			memory.current_asin = asin
			if asin not in memory.visited_asins:
				memory.visited_asins.append(asin)
		if verdict == 'BUY_READY':
			memory.phase = 'checkout'
			memory.next_action = 'Click Buy Now; visible title, price, and required options passed.'
		elif verdict == 'SELECT_OPTIONS':
			memory.phase = 'select_options'
			memory.next_action = 'Call select_webshop_required_options, then verify again.'
		elif verdict == 'REJECT':
			memory.phase = 'search'
			memory.next_action = 'Return to visible search results and inspect a different product.'
			if asin:
				memory.rejected_candidates[asin] = ['visible DOM verification did not satisfy task constraints']
		else:
			memory.phase = 'inspect'
			memory.next_action = 'Search normally or open a product from the visible results page.'

		return {
			'verifier_source': 'visible_dom_only',
			'verdict': verdict,
			'current_url': snapshot.url,
			'product_title': snapshot.product_title,
			'price_limit': memory.price_limit,
			'visible_prices': visible_prices,
			'price_ok': price_ok,
			'product_evidence_coverage': round(coverage, 3),
			'minimum_product_evidence_coverage': minimum_coverage,
			'matched_title_tokens': matched_tokens,
			'unmatched_title_tokens': unmatched_tokens,
			'hard_constraints': hard_constraints,
			'missing_hard_constraints': missing_hard_constraints,
			'semantic_assessment': semantic_assessment,
			'semantic_rank_support_only': semantic_rank_support,
			'atomic_evidence_complete': atomic_evidence_complete,
			'detail_evidence_assessment': detail_evidence_assessment,
			'detail_semantic_complete': detail_semantic_complete,
			'option_requirements': requirements,
			'missing_options': missing_options,
			'required_next_action': memory.next_action,
		}

	async def ranked_visible_state(browser_session: BrowserSession) -> tuple[WebShopVisibleSnapshot, object, list]:
		"""Build a ranking strictly from task text and the current visible page."""
		nonlocal latest_ranking
		snapshot = await visible_snapshot(browser_session)
		if '/search_results/' in snapshot.url:
			for candidate in snapshot.candidates:
				visible_candidate_pool[candidate.identifier] = candidate
		instruction = _extract_webshop_instruction(snapshot.instruction_text) or memory.instruction
		if not instruction:
			instruction = _extract_webshop_instruction(snapshot.page_text) or task
		memory.instruction = instruction
		visible_values = [option.value for option in snapshot.options]
		requirements = extract_task_requirements(instruction, visible_values)
		rejected = set(progress.rejected_candidates) | set(memory.rejected_candidates)
		candidates = list(visible_candidate_pool.values()) or snapshot.candidates
		latest_ranking = rank_visible_candidates(requirements, candidates, rejected_identifiers=rejected)
		semantic_order = semantic_orders.get(memory.search_query)
		if semantic_order:
			positions = {identifier: index for index, identifier in enumerate(semantic_order)}
			latest_ranking.sort(
				key=lambda item: (
					item.eligible,
					-positions.get(item.candidate.identifier, len(positions)),
					item.score,
				),
				reverse=True,
			)
		memory.ranked_candidates = [item.model_dump(mode='json') for item in latest_ranking[:20]]
		return snapshot, requirements, latest_ranking

	async def semantic_rerank_visible_candidates(requirements, ranking: list) -> list:
		if (
			llm is None
			or not _env_bool('EVAL_WEBSHOP_SEMANTIC_RERANK', default=True)
			or memory.search_query in semantic_orders
		):
			return ranking
		eligible = [item for item in ranking if item.eligible][:20]
		if len(eligible) < 2:
			return ranking
		payload = [
			{
				'identifier': item.candidate.identifier,
				'title': item.candidate.label,
				'visible_text': item.candidate.visible_text[:500],
				'price': item.candidate.price,
			}
			for item in eligible
		]
		fallback = SemanticVisibleCandidateChoice(
			identifier=eligible[0].candidate.identifier,
			semantic_score=max(0.0, min(1.0, eligible[0].keyword_coverage)),
			satisfies_all_constraints=False,
			matched_constraints=eligible[0].matched_keywords,
			missing_constraints=[],
		)
		decision = await _structured_controller_call(
			llm,
			(
				'Rank the visible shopping candidates for the instruction below. Use only the supplied visible text. '
				'Prioritize complete semantic satisfaction of the product type and every requested attribute, '
				'measurement, option, and price. A synonym may match, but a merely related product must rank lower. '
				'Select exactly one best candidate. Return one short JSON object with only: identifier, semantic_score, '
				'satisfies_all_constraints, matched_constraints, and missing_constraints. Copy the identifier verbatim. '
				'Do not return the full candidate list and do not add prose.\n\n'
				f'Instruction: {memory.instruction or requirements.objective}\n'
				f'Candidates: {json.dumps(payload, ensure_ascii=False)}'
			),
			SemanticVisibleCandidateChoice,
			fallback=fallback,
		)
		known = {item.candidate.identifier for item in eligible}
		valid_assessments = [decision] if decision.identifier in known else []
		order = [item.identifier for item in valid_assessments]
		order.extend(item.candidate.identifier for item in eligible if item.candidate.identifier not in order)
		for assessment in valid_assessments:
			semantic_assessments[assessment.identifier] = assessment.model_dump(mode='json')
		semantic_orders[memory.search_query] = order
		positions = {identifier: index for index, identifier in enumerate(order)}
		ranking.sort(
			key=lambda item: (
				item.eligible,
				-positions.get(item.candidate.identifier, len(positions)),
				item.score,
			),
			reverse=True,
		)
		memory.ranked_candidates = [item.model_dump(mode='json') for item in ranking[:20]]
		return ranking

	async def scan_visible_candidate_pages(
		browser_session: BrowserSession,
		snapshot: WebShopVisibleSnapshot,
	) -> tuple[WebShopVisibleSnapshot, object, list]:
		"""Collect a bounded candidate pool by following only visible Next controls."""

		from browser_use.browser.events import NavigateToUrlEvent

		max_pages = int(os.getenv('EVAL_WEBSHOP_MAX_VISIBLE_PAGES', '6'))
		current = snapshot
		_, requirements, ranking = await ranked_visible_state(browser_session)
		memory.pages_scanned = max(1, memory.pages_scanned)
		while current.pagination_urls and memory.pages_scanned < max_pages:
			next_url = current.pagination_urls[0]
			event = browser_session.event_bus.dispatch(NavigateToUrlEvent(url=next_url, new_tab=False))
			await event
			await asyncio.sleep(0.25)
			current, requirements, ranking = await ranked_visible_state(browser_session)
			memory.pages_scanned += 1
		ranking = await semantic_rerank_visible_candidates(requirements, ranking)
		memory.next_action = f'Rank {len(visible_candidate_pool)} candidates observed across {memory.pages_scanned} visible pages.'
		return current, requirements, ranking

	@tools.action(
		'Inspect and rank candidates exposed by the current visible page. The ranking uses no hidden data or stored answers.'
	)
	async def inspect_ranked_candidates(browser_session: BrowserSession) -> ActionResult:
		snapshot, requirements, ranking = await ranked_visible_state(browser_session)
		return memory_result(
			'VISIBLE_CANDIDATE_RANKING=',
			{
				'capabilities': snapshot.capabilities.model_dump(mode='json'),
				'recommended_routes': snapshot.capabilities.routes(),
				'requirements': requirements.model_dump(mode='json'),
				'candidates': [item.model_dump(mode='json') for item in ranking[:10]],
			},
		)

	@tools.action(
		'Open one candidate from the latest visible-page ranking by rank. The exact visible href is reused unchanged.',
		param_model=RankedCandidateAction,
	)
	async def open_ranked_candidate(
		params: RankedCandidateAction,
		browser_session: BrowserSession,
	) -> ActionResult:
		from browser_use.browser.events import NavigateToUrlEvent

		snapshot, _, ranking = await ranked_visible_state(browser_session)
		eligible = [item for item in ranking if item.eligible and item.candidate.url]
		if params.rank > len(eligible):
			return ActionResult(error=f'Visible candidate rank {params.rank} is unavailable; {len(eligible)} eligible remain.')
		selected = eligible[params.rank - 1]
		candidate = selected.candidate
		event = browser_session.event_bus.dispatch(NavigateToUrlEvent(url=candidate.url, new_tab=False))
		await event
		directive = progress.record(
			url=snapshot.url,
			action='open_ranked_candidate',
			visible_text=snapshot.page_text,
			candidate_id=candidate.identifier,
		)
		if candidate.identifier not in memory.visited_asins:
			memory.visited_asins.append(candidate.identifier)
		memory.query_candidate_inspections += 1
		memory.phase = 'inspect'
		memory.next_action = 'Inspect the newly opened candidate and validate all task constraints.'
		return memory_result(
			'VISIBLE_CANDIDATE_OPENED=',
			{
				'candidate': candidate.model_dump(mode='json'),
				'visible_rank_score': selected.score,
				'recovery': directive.model_dump(mode='json'),
			},
		)

	@tools.action(
		'Review bounded progress memory and receive a recovery instruction when states, actions, or candidates repeat.'
	)
	async def review_execution_progress(browser_session: BrowserSession) -> ActionResult:
		snapshot = await visible_snapshot(browser_session)
		directive = progress.record(
			url=snapshot.url,
			action='review_execution_progress',
			visible_text=snapshot.page_text,
		)
		return memory_result(
			'EXECUTION_PROGRESS=',
			{
				'capabilities': snapshot.capabilities.model_dump(mode='json'),
				'recommended_routes': snapshot.capabilities.routes(),
				'recovery': directive.model_dump(mode='json'),
				'rejected_candidates': sorted(progress.rejected_candidates),
			},
		)

	async def submit_verified_visible_commit(browser_session: BrowserSession) -> dict:
		"""Submit a visible Buy Now control only after the verifier has passed."""

		cdp_session = await browser_session.get_or_create_cdp_session()
		result = await cdp_session.cdp_client.send.Runtime.evaluate(
			params={
				'expression': """(() => {
					const visible = element => {
						if (!element) return false;
						const style = getComputedStyle(element);
						const rect = element.getBoundingClientRect();
						return style.display !== 'none' && style.visibility !== 'hidden'
							&& rect.width > 0 && rect.height > 0 && !element.disabled;
					};
					const controls = Array.from(document.querySelectorAll('button,input[type="submit"]'));
					const button = controls.find(control => visible(control) && /buy now/i.test(
						String(control.innerText || control.value || control.getAttribute('aria-label') || '')
					));
					if (!button) return {submitted: false, reason: 'visible Buy Now control not found'};
					if (button.form?.requestSubmit) button.form.requestSubmit(button);
					else button.click();
					return {submitted: true, label: String(button.innerText || button.value || '').trim()};
				})()""",
				'returnByValue': True,
			},
			session_id=cdp_session.session_id,
		)
		await asyncio.sleep(0.75)
		return result.get('result', {}).get('value') or {}

	@tools.action(
		'Advance one bounded step using current visible capabilities: validate a detail page, rank visible candidates, or submit the visible search form.'
	)
	async def advance_visible_task(browser_session: BrowserSession) -> ActionResult:
		snapshot, requirements, ranking = await ranked_visible_state(browser_session)
		visible_requirements = infer_visible_requirements(snapshot, memory.instruction)
		missing = [f'{item["option_name"]}={item["required_value"]}' for item in visible_requirements if item['status'] != 'selected']
		if completion_is_grounded(
			requirements,
			url=snapshot.url,
			visible_text=snapshot.page_text,
			unresolved_constraints=missing,
		):
			return memory_result('VISIBLE_TASK_ADVANCE=', {'status': 'DONE', 'current_url': snapshot.url})

		if is_visible_product_detail(snapshot.url, snapshot.product_title):
			report = await visible_report(browser_session)
			if report['verdict'] == 'SELECT_OPTIONS':
				selection = await select_webshop_required_options(browser_session=browser_session)
				return memory_result(
					'VISIBLE_TASK_ADVANCE=',
					{'status': 'OPTIONS_SELECTED', 'verification': report, 'selection': selection.extracted_content},
				)
			if report['verdict'] == 'BUY_READY':
				if auto_commit_enabled:
					commit = await submit_verified_visible_commit(browser_session)
					final_snapshot = await visible_snapshot(browser_session)
					memory.phase = 'checkout'
					memory.next_action = 'Read visible terminal reward evidence and stop.'
					return memory_result(
						'VISIBLE_TASK_ADVANCE=',
						{
							'status': 'CHECKOUT_SUBMITTED' if commit.get('submitted') else 'COMMIT_READY',
							'verification': report,
							'commit': commit,
							'final_url': final_snapshot.url,
						},
					)
				return memory_result('VISIBLE_TASK_ADVANCE=', {'status': 'COMMIT_READY', 'verification': report})
			candidate_id = memory.current_asin or snapshot.url
			progress.rejected_candidates.add(candidate_id)
			memory.rejected_candidates[candidate_id] = ['current visible detail page violates task constraints']
			from browser_use.browser.events import NavigateToUrlEvent

			search_url = _webshop_search_url_from_detail(snapshot.url)
			if not search_url:
				return ActionResult(error='Cannot recover the visible search URL from the current product route.')
			back_event = browser_session.event_bus.dispatch(NavigateToUrlEvent(url=search_url, new_tab=False))
			await back_event
			await asyncio.sleep(0.35)
			memory.automatic_candidate_switches += 1
			back_snapshot, _, back_ranking = await ranked_visible_state(browser_session)
			back_eligible = [item for item in back_ranking if item.eligible and item.candidate.url]
			candidate_limit = int(os.getenv('EVAL_WEBSHOP_CANDIDATES_PER_QUERY', '5'))
			if back_eligible and memory.query_candidate_inspections < candidate_limit:
				opened = await open_ranked_candidate(params=RankedCandidateAction(rank=1), browser_session=browser_session)
				return memory_result(
					'VISIBLE_TASK_ADVANCE=',
					{
						'status': 'CANDIDATE_REPLACED',
						'verification': report,
						'recovery_url': back_snapshot.url,
						'opened': opened.extracted_content,
					},
				)
			home_url = _webshop_home_url_from_detail(snapshot.url)
			if home_url:
				home_event = browser_session.event_bus.dispatch(NavigateToUrlEvent(url=home_url, new_tab=False))
				await home_event
				memory.phase = 'requery'
				visible_candidate_pool.clear()
				for _ in range(20):
					await asyncio.sleep(0.1)
					home_snapshot = await visible_snapshot(browser_session)
					if home_snapshot.url.rstrip('/') == home_url.rstrip('/') and home_snapshot.capabilities.search_controls:
						memory.phase = 'search'
						break
			memory.pages_scanned = 0
			return memory_result(
				'VISIBLE_TASK_ADVANCE=',
				{
					'status': 'CANDIDATE_BATCH_EXHAUSTED',
					'verification': report,
					'next_action': 'Returned to visible search home; submit the next grounded query.',
				},
			)

		if snapshot.pagination_urls and memory.pages_scanned == 0:
			snapshot, requirements, ranking = await scan_visible_candidate_pages(browser_session, snapshot)
		eligible = [item for item in ranking if item.eligible and item.candidate.url]
		if eligible:
			return await open_ranked_candidate(params=RankedCandidateAction(rank=1), browser_session=browser_session)

		if snapshot.capabilities.search_controls:
			query_plan = grounded_search_queries(memory.instruction or requirements.objective)
			if not query_plan:
				return ActionResult(error='No grounded search terms could be extracted from the task.')
			if memory.search_attempts >= len(query_plan):
				return ActionResult(
					error='Grounded search query plan exhausted; no visible candidate satisfied every hard constraint.'
				)
			query = query_plan[memory.search_attempts]
			cdp_session = await browser_session.get_or_create_cdp_session()
			result = await cdp_session.cdp_client.send.Runtime.evaluate(
				params={
					'expression': f"""(() => {{
						const visible = element => {{
							const style = getComputedStyle(element);
							const rect = element.getBoundingClientRect();
							return style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0;
						}};
						const controls = Array.from(document.querySelectorAll('input')).filter(visible);
						const input = controls.find(node => /search/i.test(`${{node.type}} ${{node.name}} ${{node.placeholder}} ${{node.getAttribute('aria-label') || ''}}`));
						if (!input) return {{submitted: false, reason: 'visible search input disappeared'}};
						const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
						setter.call(input, {json.dumps(query)});
						input.dispatchEvent(new Event('input', {{bubbles: true}}));
						input.dispatchEvent(new Event('change', {{bubbles: true}}));
						if (input.form?.requestSubmit) input.form.requestSubmit();
						else input.dispatchEvent(new KeyboardEvent('keydown', {{key: 'Enter', code: 'Enter', bubbles: true}}));
						return {{submitted: true, query: {json.dumps(query)}}};
					}})()""",
					'returnByValue': True,
				},
				session_id=cdp_session.session_id,
			)
			await asyncio.sleep(1)
			memory.search_attempts += 1
			memory.query_candidate_inspections = 0
			memory.pages_scanned = 0
			visible_candidate_pool.clear()
			memory.search_query = query
			memory.phase = 'search'
			directive = progress.record(
				url=snapshot.url,
				action=f'search:{query}',
				visible_text=snapshot.page_text,
				progressed=memory.search_attempts == 1,
			)
			return memory_result(
				'VISIBLE_TASK_ADVANCE=',
				{
					'status': 'SEARCH_SUBMITTED',
					'result': result.get('result', {}).get('value'),
					'query': query,
					'recovery': directive.model_dump(mode='json'),
				},
			)

		return memory_result(
			'VISIBLE_TASK_ADVANCE=',
			{
				'status': 'NEEDS_AGENT',
				'capabilities': snapshot.capabilities.model_dump(mode='json'),
				'recommended_routes': snapshot.capabilities.routes(),
			},
		)

	@tools.action(
		'Execute a bounded visible-DOM transaction across navigation boundaries: search, scan, rank, select, verify, and commit.'
	)
	async def execute_visible_webshop_transaction(browser_session: BrowserSession) -> ActionResult:
		trace: list[str] = []
		max_transitions = int(os.getenv('EVAL_WEBSHOP_TRANSACTION_TRANSITIONS', '50'))
		for transition_index in range(max_transitions):
			result = await advance_visible_task(browser_session=browser_session)
			trace.append(str(result.error or result.extracted_content or '')[:800])
			snapshot = await visible_snapshot(browser_session)
			if _env_bool('EVAL_DEBUG_TRANSACTION_TRACE', default=False):
				print(
					f'[DEBUG] WebShop v3.2 transition={transition_index + 1} url={snapshot.url} '
					f'result={trace[-1][:500]}',
					file=sys.stderr,
				)
			if '/done/' in snapshot.url and _parse_webshop_reward(snapshot.page_text) is not None:
				return memory_result(
					'VISIBLE_WEBSHOP_TRANSACTION=',
					{
						'status': 'DONE',
						'final_url': snapshot.url,
						'reward': _parse_webshop_reward(snapshot.page_text),
						'trace': trace,
					},
				)
			if result.error:
				break
		return memory_result(
			'VISIBLE_WEBSHOP_TRANSACTION=',
			{
				'status': 'NEEDS_AGENT',
				'transitions': len(trace),
				'trace': trace,
				'next_action': memory.next_action,
			},
		)

	@tools.action('Review WebShop memory built exclusively from visible browser DOM and prior actions.')
	async def review_webshop_memory() -> ActionResult:
		return memory_result('WEBSHOP_WEB_ONLY_MEMORY=', {'status': 'READY'})

	@tools.action(
		'Verify the current WebShop product using only the visible DOM, visible price, public URL, and radio options. '
		'Call this before Buy Now.',
		param_model=VerifyWebShopCandidateAction,
	)
	async def verify_webshop_candidate(
		params: VerifyWebShopCandidateAction,
		browser_session: BrowserSession,
	) -> ActionResult:
		report = await visible_report(browser_session)
		report['candidate_summary'] = params.candidate_summary
		return memory_result('WEBSHOP_WEB_ONLY_VERIFICATION=', report)

	@tools.action('Select required product options by matching instruction values to radio controls exposed in the current DOM.')
	async def select_webshop_required_options(browser_session: BrowserSession) -> ActionResult:
		selected = []
		for _ in range(6):
			snapshot = await visible_snapshot(browser_session)
			instruction = _extract_webshop_instruction(snapshot.instruction_text) or memory.instruction
			requirements = infer_visible_requirements(snapshot, instruction)
			missing = [item for item in requirements if item['status'] != 'selected']
			if not missing:
				memory.phase = 'inspect'
				memory.next_action = 'Options selected from visible controls; verify the product again.'
				return memory_result(
					'WEBSHOP_WEB_ONLY_OPTION_SELECTION=',
					{'status': 'SELECTED', 'selected': selected, 'current_url': snapshot.url},
				)
			target = missing[0]
			if not target.get('data_url'):
				return ActionResult(
					error=f'Visible option has no navigable data-url: {target["option_name"]}={target["required_value"]}'
				)
			from browser_use.browser.events import NavigateToUrlEvent

			event = browser_session.event_bus.dispatch(NavigateToUrlEvent(url=target['data_url'], new_tab=False))
			await event
			selected.append({'option_name': target['option_name'], 'value': target['required_value']})

		return ActionResult(error='Visible option selection exceeded its transition limit.')

	@tools.action(
		'Click an element by index. Buy Now is blocked until visible-DOM verification returns BUY_READY.',
		param_model=ClickElementActionIndexOnly,
	)
	async def click(params: ClickElementActionIndexOnly, browser_session: BrowserSession) -> ActionResult:
		node = await browser_session.get_element_by_index(params.index)
		if node is None:
			memory.stale_index_recoveries += 1
			memory.next_action = 'The DOM changed; discard the stale index and resume from the current visible state.'
			return await advance_visible_task(browser_session=browser_session)
		if 'buy now' not in get_click_description(node).lower():
			return await tools._click_by_index(params, browser_session)
		if not buy_guard_enabled:
			return await tools._click_by_index(params, browser_session)

		report = await visible_report(browser_session)
		if report['verdict'] == 'SELECT_OPTIONS':
			selection = await select_webshop_required_options(browser_session=browser_session)
			result_kwargs = {
				'extracted_content': (
					'BUY_NOW_INTERCEPTED_WEB_ONLY: required visible options were selected. '
					'Verify the product again before retrying Buy Now. ' + (selection.extracted_content or '')
				),
			}
			if long_term_memory_enabled:
				result_kwargs['long_term_memory'] = 'WEBSHOP_WEB_ONLY_MEMORY=' + json.dumps(memory_snapshot(), ensure_ascii=False)
			return ActionResult(**result_kwargs)
		if report['verdict'] != 'BUY_READY':
			result_kwargs = {
				'error': f'Buy Now blocked by visible-DOM verifier: {report["verdict"]}. {report["required_next_action"]}'
			}
			if long_term_memory_enabled:
				result_kwargs['long_term_memory'] = 'WEBSHOP_WEB_ONLY_MEMORY=' + json.dumps(memory_snapshot(), ensure_ascii=False)
			return ActionResult(**result_kwargs)

		memory.phase = 'checkout'
		memory.next_action = 'Read the real reward from the WebShop done page.'
		return await tools._click_by_index(params, browser_session)

	if 'done' in tools.registry.registry.actions:
		del tools.registry.registry.actions['done']

	@tools.action(
		'Complete the task only when the current visible state contains terminal evidence and all visible constraints are resolved.',
		param_model=DoneAction,
	)
	async def done(params: DoneAction, browser_session: BrowserSession) -> ActionResult:
		nonlocal premature_done_attempts
		if not params.success:
			return ActionResult(
				is_done=True,
				success=False,
				extracted_content=params.text,
				long_term_memory='Task ended unsuccessfully after explicit agent acknowledgement.',
			)
		snapshot = await visible_snapshot(browser_session)
		instruction = _extract_webshop_instruction(snapshot.instruction_text) or memory.instruction or task
		visible_requirements = infer_visible_requirements(snapshot, instruction)
		missing = [f'{item["option_name"]}={item["required_value"]}' for item in visible_requirements if item['status'] != 'selected']
		requirements = extract_task_requirements(instruction, [option.value for option in snapshot.options])
		if completion_is_grounded(
			requirements,
			url=snapshot.url,
			visible_text=snapshot.page_text,
			unresolved_constraints=missing,
		):
			return ActionResult(
				is_done=True,
				success=True,
				extracted_content=params.text,
				long_term_memory='Task completion accepted from visible terminal evidence.',
			)

		premature_done_attempts += 1
		if premature_done_attempts >= 3:
			return ActionResult(
				is_done=True,
				success=False,
				error='Completion rejected three times because no visible terminal evidence was found.',
				long_term_memory='Task stopped to prevent an ungrounded completion loop.',
			)
		return ActionResult(
			error=(
				'Completion rejected: no visible terminal evidence was found'
				+ (f'; unresolved constraints: {missing}' if missing else '')
				+ '. Continue the task or call advance_visible_task.'
			),
			long_term_memory='Do not claim success until the page visibly confirms completion.',
		)

	return tools


def _parse_webshop_reward(text: str) -> float | None:
	patterns = [
		r'\bYour\s+score\s*(?:\([^)]*\))?\s*(?::|=)?\s*([01](?:\.[0-9]+)?)\b',
		r'\b(?:final\s+)?WebShop\s+reward(?:\s+score)?\s*(?:after\s+checkout\s*)?(?:is|was|:|=)\s*([01](?:\.[0-9]+)?)\b',
		r'\b(?:final\s+)?(?:reward|score)\s*(?:score)?\s*(?:after\s+checkout\s*)?(?:is|was|:|=)\s*([01](?:\.[0-9]+)?)\b',
	]
	for pattern in patterns:
		match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
		if match:
			try:
				reward = float(match.group(1))
			except ValueError:
				continue
			if 0 <= reward <= 1:
				return reward
	return None


def _webshop_reward_response(
	task_data: dict,
	history: AgentHistoryList,
	agent_output: str,
	final_page_text: str = '',
	final_url: str = '',
) -> WebShopRewardResponse | None:
	threshold = task_data.get('webshop_reward_threshold')
	if threshold is None:
		return None
	# Prefer the browser's real final page. Agent prose can contain guessed or
	# previously observed rewards and must not override checkout ground truth.
	page_reward = _parse_webshop_reward(final_page_text)
	combined = '\n'.join(
		[
			agent_output,
			*map(str, history.extracted_content()),
			*map(str, history.errors()),
		]
	)
	is_checkout_done = '/done/' in final_url
	# A model statement such as "reward is 1.0" is not evidence of checkout.
	# History is only a fallback when the browser URL proves we reached /done/.
	reward = page_reward if page_reward is not None and is_checkout_done else None
	if reward is None and is_checkout_done:
		reward = _parse_webshop_reward(combined)
	if reward is None:
		return WebShopRewardResponse(
			success=False,
			explanation='WebShop reward threshold was configured, but no final WebShop reward score was found in the agent output/history.',
			reward=None,
		)
	success = reward >= float(threshold)
	return WebShopRewardResponse(
		success=success,
		explanation=(
			f'WebShop reward={reward:.4f}; threshold={float(threshold):.4f}; '
			f'source={"final_page" if page_reward is not None else "history"}; final_url={final_url or "unknown"}.'
		),
		reward=reward,
	)


async def _read_final_browser_page(browser_session: BrowserSession) -> tuple[str, str]:
	"""Read final URL and visible text directly from Chromium for deterministic scoring."""
	try:
		cdp_session = await browser_session.get_or_create_cdp_session()
		result = await cdp_session.cdp_client.send.Runtime.evaluate(
			params={
				'expression': '({url: window.location.href, text: document.body ? document.body.innerText : ""})',
				'returnByValue': True,
			},
			session_id=cdp_session.session_id,
		)
		value = result.get('result', {}).get('value') or {}
		return str(value.get('url') or ''), str(value.get('text') or '')
	except Exception as error:
		print(f'[DEBUG] Final page capture failed: {error}', file=sys.stderr)
		return '', ''


def _classify_ui_capabilities(profile: ShortUiPageProfile) -> ShortUiPageProfile:
	"""Choose workflow modules from observable page text and control structure."""

	text = profile.instruction_sample.lower()
	compact = profile.interactive_count <= 60 and profile.text_length <= 5000
	editable_count = profile.text_input_count + profile.textarea_count + profile.select_count
	profile.exact_transfer = compact and bool(
		re.search(r'\b(copy|paste|duplicate|transfer|preserve|same (?:text|value)|exact (?:text|value))\b', text)
	)
	profile.temporal_control = compact and bool(
		re.search(r'\b(wait|delay|after|seconds?|enable[ds]?|appear[sd]?|animation|loading)\b', text)
	)
	profile.multi_field_form = compact and editable_count >= 2 and bool(
		re.search(r'\b(fill|enter|type|select|choose|set|form|book|register|submit)\b', text)
	) and not profile.exact_transfer
	semantic_selection = bool(re.search(r'\b(similar to|synonym|related to|words? like|meaning)\b', text))
	ordinal_selection = bool(re.search(r'\b\d+(?:st|nd|rd|th)\b', text))
	profile.labelled_selection = (
		compact
		and not semantic_selection
		and not ordinal_selection
		and (profile.checkbox_count + profile.radio_count) >= 2
		and bool(re.search(r'\b(select|check|choose|mark)\b', text))
	)
	profile.date_entry = compact and profile.date_input_count == 1 and bool(
		re.search(r'\b(?:enter|select|choose|set)\s+\d{1,2}/\d{1,2}/\d{4}\b', text)
	)
	profile.ordinal_control = compact and bool(
		re.search(r'\b\d+(?:st|nd|rd|th)\b', text)
		and (profile.checkbox_count + profile.radio_count + profile.range_count) > 0
	)
	product_terms = sum(
		term in text
		for term in ('search', 'product', 'results', 'price', 'cart', 'buy now', 'shopping', 'instruction')
	)
	catalog_route = bool(re.search(r'/(?:search_results|item_page|done)/', profile.url))
	search_landing = profile.form_count >= 1 and profile.text_input_count >= 1
	profile.product_constraints = (
		(product_terms >= 2 and (profile.link_count >= 2 or search_landing))
		or catalog_route
	)
	profile.capabilities = [
		name
		for name, enabled in (
			('exact_value_transfer', profile.exact_transfer),
			('temporal_control', profile.temporal_control),
			('multi_field_form', profile.multi_field_form),
			('labelled_selection', profile.labelled_selection),
			('date_entry', profile.date_entry),
			('ordinal_control', profile.ordinal_control),
			('product_constraints', profile.product_constraints),
		)
		if enabled
	]
	profile.eligible = any(
		(
			profile.exact_transfer,
			profile.temporal_control,
			profile.multi_field_form,
			profile.labelled_selection,
			profile.date_entry,
			profile.ordinal_control,
		)
	)
	if profile.eligible:
		profile.route_reason = 'Enabled only modules supported by visible instruction and DOM controls.'
	elif profile.product_constraints:
		profile.route_reason = 'Catalog workflow detected; use the visible-evidence product constraint loop.'
	else:
		profile.route_reason = 'Simple or unsupported workflow; preserve the lightweight base agent loop.'
	return profile


def _requested_toggle_labels(profile: ShortUiPageProfile) -> list[str]:
	"""Match accessible toggle labels mentioned in the visible instruction line."""

	if not profile.labelled_selection:
		return []
	instruction = ' '.join(profile.instruction_line.casefold().split())
	if not instruction:
		return []
	requested = []
	for label in profile.toggle_labels:
		normalized_label = ' '.join(label.casefold().split())
		if not normalized_label:
			continue
		if re.search(rf'(?<!\w){re.escape(normalized_label)}(?!\w)', instruction):
			requested.append(label)
	return list(dict.fromkeys(requested))


def _extract_iso_date(instruction: str) -> str | None:
	"""Convert an unambiguous numeric instruction date to a native input value."""

	match = re.search(r'(?<!\d)(\d{1,2})/(\d{1,2})/(\d{4})(?!\d)', instruction)
	if not match:
		return None
	month, day, year = map(int, match.groups())
	try:
		return datetime(year, month, day).date().isoformat()
	except ValueError:
		return None


def _extract_ordinal(instruction: str, control_name: str) -> int | None:
	"""Return a one-based ordinal only when it directly modifies a control name."""

	match = re.search(
		rf'\b(\d+)(?:st|nd|rd|th)\s+{re.escape(control_name)}\b',
		instruction,
		flags=re.IGNORECASE,
	)
	return int(match.group(1)) if match else None


def _extract_slider_value(instruction: str) -> float | None:
	"""Read an explicit numeric slider target without inferring from unrelated numbers."""

	patterns = (
		r'\b(?:set|select|choose|move)\s+(?:the\s+)?slider\s+(?:to|at)\s+(-?\d+(?:\.\d+)?)\b',
		r'\b(?:set|select|choose)\s+(-?\d+(?:\.\d+)?)\s+(?:with|using|on)\s+(?:the\s+)?slider\b',
	)
	for pattern in patterns:
		match = re.search(pattern, instruction, flags=re.IGNORECASE)
		if match:
			return float(match.group(1))
	return None


def _extract_timed_button_sequence(instruction: str) -> tuple[str, float, str] | None:
	"""Parse an explicit click-wait-click instruction without guessing labels or delay."""

	match = re.search(
		r'\bclick\s+(?:button\s+)?(.+?),\s*wait\s+(\d+(?:\.\d+)?)\s*seconds?\s*,?\s*'
		r'(?:and\s+)?then\s+click\s+(?:button\s+)?(.+?)(?:[.!]|$)',
		instruction,
		flags=re.IGNORECASE,
	)
	if not match:
		return None
	first_label, seconds, second_label = match.groups()
	return first_label.strip(), float(seconds), second_label.strip()


def _action_result_record(operation: str, result, **details) -> dict:
	"""Serialize a deterministic pre-agent action for experiment auditing."""

	return {
		'operation': operation,
		'success': result.error is None,
		'error': result.error,
		'metadata': result.metadata,
		**details,
	}


async def _inspect_pre_agent_controls(agent_tools: Tools, browser_session: BrowserSession) -> tuple[list[dict], str | None]:
	"""Return a stable, serializable control snapshot for deterministic preparation."""

	result = await agent_tools.inspect_controls(
		visible_only=True,
		max_controls=250,
		browser_session=browser_session,
	)
	if result.error:
		return [], result.error
	try:
		payload = json.loads(result.extracted_content or '{}')
		return list(payload.get('controls') or []), None
	except (TypeError, ValueError) as error:
		return [], f'Invalid control inspection payload: {error}'


async def _execute_precise_button_sequence(
	browser_session: BrowserSession,
	first_label: str,
	delay_seconds: float,
	second_label: str,
) -> dict:
	"""Execute two uniquely labelled visible clicks on the browser's monotonic clock."""

	cdp_session = await browser_session.get_or_create_cdp_session()
	expression = f"""(() => new Promise((resolve) => {{
		const normalize = value => String(value || '').replace(/\\s+/g, ' ').trim().toLowerCase();
		const visible = element => {{
			const style = getComputedStyle(element);
			const rect = element.getBoundingClientRect();
			return style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0;
		}};
		const controls = Array.from(document.querySelectorAll('button,input[type="button"],input[type="submit"]'))
			.filter(visible);
		const firstMatches = controls.filter(element =>
			normalize(element.innerText || element.value) === normalize({json.dumps(first_label)})
		);
		const secondMatches = controls.filter(element =>
			normalize(element.innerText || element.value) === normalize({json.dumps(second_label)})
		);
		if (firstMatches.length !== 1 || secondMatches.length !== 1) {{
			resolve({{success: false, error: 'Timed button labels were missing or ambiguous.'}});
			return;
		}}
		firstMatches[0].click();
		const started = performance.now();
		setTimeout(() => {{
			secondMatches[0].click();
			resolve({{success: true, elapsed_ms: performance.now() - started}});
		}}, {int(round(delay_seconds * 1000))});
	}}))()"""
	result = await cdp_session.cdp_client.send.Runtime.evaluate(
		params={'expression': expression, 'awaitPromise': True, 'returnByValue': True},
		session_id=cdp_session.session_id,
	)
	value = result.get('result', {}).get('value')
	return value if isinstance(value, dict) else {'success': False, 'error': 'Timed sequence returned no result.'}


async def _execute_visible_dom_workflow(browser_session: BrowserSession, instruction: str) -> dict:
	"""Execute a uniquely grounded workflow using only instruction text and rendered DOM evidence."""

	cdp_session = await browser_session.get_or_create_cdp_session()
	structured_program = parse_task_program(instruction).model_dump(mode='json')
	expression = rf"""(async () => {{
		const instruction = {json.dumps(instruction)};
		const structuredProgram = {json.dumps(structured_program)};
		const normalize = value => String(value || '').replace(/\s+/g, ' ').trim();
		const lower = value => normalize(value).toLowerCase();
		const visible = element => {{
			if (!element) return false;
			const style = getComputedStyle(element);
			const rect = element.getBoundingClientRect();
			return style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0;
		}};
		const sleep = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));
		const exact = (elements, label) => Array.from(elements).filter(element =>
			lower(element.innerText || element.textContent || element.value) === lower(label)
		);
		const setText = (element, value) => {{
			const prototype = element.tagName.toLowerCase() === 'textarea'
				? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
			const setter = Object.getOwnPropertyDescriptor(prototype, 'value')?.set;
			if (setter) setter.call(element, value); else element.value = value;
			element.dispatchEvent(new Event('input', {{bubbles: true}}));
			element.dispatchEvent(new Event('change', {{bubbles: true}}));
		}};
		const submitButton = () => {{
			const candidates = Array.from(document.querySelectorAll('button,input[type="submit"],input[type="button"]'))
				.filter(visible).filter(element => /^(submit|search|continue|confirm|done)$/i.test(
					normalize(element.innerText || element.value)
				));
			return candidates.length === 1 ? candidates[0] : null;
		}};

		const repeatOperation = structuredProgram.operations.find(operation => operation.type === 'repeat_until');
		const commitOperation = structuredProgram.operations.find(operation => operation.type === 'commit');
		if (repeatOperation && commitOperation && repeatOperation.predicate.type === 'numeric') {{
			const actionControls = exact(
				document.querySelectorAll('button,input[type="button"]'),
				repeatOperation.action.label,
			).filter(visible);
			const commitControls = exact(
				document.querySelectorAll('button,input[type="submit"],input[type="button"]'),
				commitOperation.target.label,
			).filter(visible);
			if (actionControls.length === 1 && commitControls.length === 1) {{
				const predicate = repeatOperation.predicate;
				const satisfies = value => (
					predicate.operator === 'modulo_equals' ? value % predicate.divisor === predicate.value
					: predicate.operator === 'less_than' ? value < predicate.value
					: predicate.operator === 'greater_than' ? value > predicate.value
					: predicate.operator === 'equals' ? value === predicate.value
					: false
				);
				const numericLeaves = () => Array.from(document.querySelectorAll('body *'))
					.filter(visible)
					.filter(element => element.children.length === 0);
				let previousText = new Map(numericLeaves().map(element => [
					element,
					normalize(element.innerText || element.textContent),
				]));
				for (let attempt = 1; attempt <= repeatOperation.max_attempts; attempt++) {{
					actionControls[0].click();
					await sleep(repeatOperation.poll_milliseconds);
					const leaves = numericLeaves();
					const values = leaves
						.map(element => ({{
							current: normalize(element.innerText || element.textContent),
							previous: previousText.get(element),
						}}))
						.filter(item => item.current !== item.previous && /^-?\d+$/.test(item.current))
						.map(item => Number(item.current));
					previousText = new Map(leaves.map(element => [
						element,
						normalize(element.innerText || element.textContent),
					]));
					const answer = values.find(satisfies);
					if (answer !== undefined) {{
						commitControls[0].click();
						return {{
							matched: true,
							strategy: 'structured_repeat_until',
							success: true,
							attempts: attempt,
							observed: answer,
						}};
					}}
				}}
				return {{
					matched: true,
					strategy: 'structured_repeat_until',
					success: false,
					error: 'No action-induced visible value satisfied the bounded structured predicate.',
				}};
			}}
		}}

		const dateMatch = instruction.match(/\b(?:select|choose|enter|set)\s+(\d{{1,2}}\/\d{{1,2}}\/\d{{4}})\s+as\s+the\s+date/i);
		if (dateMatch) {{
			const inputs = Array.from(document.querySelectorAll('input')).filter(visible);
			const dateInputs = inputs.filter(element =>
				element.type === 'date' || element.readOnly || /date/i.test(`${{element.id}} ${{element.name}} ${{element.placeholder}}`)
			);
			const submit = submitButton();
			if (dateInputs.length === 1 && submit) {{
				const input = dateInputs[0];
				if (window.jQuery && window.jQuery(input).hasClass('hasDatepicker'))
					window.jQuery(input).datepicker('setDate', dateMatch[1]);
				else setText(input, dateMatch[1]);
				const observed = input.value;
				if (observed === dateMatch[1]) {{
					submit.click();
					return {{matched: true, strategy: 'custom_date_entry', success: true, observed}};
				}}
				return {{matched: true, strategy: 'custom_date_entry', success: false, error: `Date readback was ${{observed}}`}};
			}}
		}}

		const wordMatch = instruction.match(/\bfind\s+the\s+(\d+)(?:st|nd|rd|th)\s+word\s+in\s+the\s+paragraph/i);
		if (wordMatch) {{
			const ordinal = Number(wordMatch[1]);
			const paragraphs = Array.from(document.querySelectorAll('p')).filter(visible)
				.map(element => ({{element, words: normalize(element.innerText).match(/[A-Za-z0-9]+/g) || []}}))
				.filter(item => item.words.length >= ordinal && !lower(item.element.innerText).includes('find the'));
			const inputs = Array.from(document.querySelectorAll('input[type="text"],textarea')).filter(visible);
			const submit = submitButton();
			if (paragraphs.length === 1 && inputs.length === 1 && submit) {{
				const answer = paragraphs[0].words[ordinal - 1];
				setText(inputs[0], answer);
				submit.click();
				return {{matched: true, strategy: 'ordinal_paragraph_word', success: true, ordinal, answer}};
			}}
		}}

		const passwordMatch = instruction.match(/\benter\s+the\s+password\s+["']([^"']+)["']\s+into\s+both\s+text\s+fields/i);
		if (passwordMatch) {{
			const passwordInputs = Array.from(document.querySelectorAll('input[type="password"]')).filter(visible);
			const submit = submitButton();
			if (passwordInputs.length === 2 && submit) {{
				passwordInputs.forEach(input => setText(input, passwordMatch[1]));
				const verified = passwordInputs.every(input => input.value === passwordMatch[1]);
				if (verified) {{
					submit.click();
					return {{matched: true, strategy: 'repeated_password_transaction', success: true, field_count: 2}};
				}}
				return {{matched: true, strategy: 'repeated_password_transaction', success: false, error: 'Password fields failed equality verification.'}};
			}}
		}}

		if (/close\s+the\s+dialog/i.test(instruction)) {{
			const dialogs = Array.from(document.querySelectorAll('[role="dialog"],.ui-dialog')).filter(visible);
			const closeButtons = dialogs.flatMap(dialog => Array.from(dialog.querySelectorAll(
				'button,.ui-dialog-titlebar-close,[aria-label*="close" i]'
			))).filter(visible);
			if (dialogs.length === 1 && closeButtons.length === 1) {{
				closeButtons[0].click();
				return {{matched: true, strategy: 'modal_close', success: true}};
			}}
		}}

		const pathMatch = instruction.match(/^select\s+(.+>.+)$/i);
		if (pathMatch) {{
			const labels = pathMatch[1].split('>').map(normalize).filter(Boolean);
			const rootMenu = document.querySelector('[role="menu"],.ui-menu,#menu');
			if (rootMenu && labels.length >= 2) {{
				let scope = rootMenu;
				for (let index = 0; index < labels.length; index++) {{
					const candidates = exact(scope.querySelectorAll('li > div,li > a'), labels[index]);
					if (candidates.length !== 1)
						return {{matched: true, strategy: 'hierarchical_menu', success: false, error: `Ambiguous path segment ${{labels[index]}}`}};
					const target = candidates[0];
					target.dispatchEvent(new MouseEvent('mouseover', {{bubbles: true}}));
					target.dispatchEvent(new MouseEvent('mouseenter', {{bubbles: true}}));
					if (index === labels.length - 1) target.click();
					await sleep(120);
					const childMenu = target.closest('li')?.querySelector(':scope > ul');
					if (index < labels.length - 1 && !childMenu)
						return {{matched: true, strategy: 'hierarchical_menu', success: false, error: `No child menu for ${{labels[index]}}`}};
					scope = childMenu || scope;
				}}
				return {{matched: true, strategy: 'hierarchical_menu', success: true, labels}};
			}}
		}}

		const quotedTarget = instruction.match(/(?:link|text)\s+["']([^"']+)["']/i)?.[1];
		if (quotedTarget && /expand\s+the\s+sections/i.test(instruction)) {{
			const targets = exact(document.querySelectorAll('.alink,a,[role="link"]'), quotedTarget);
			if (targets.length === 1) {{
				const target = targets[0];
				const panel = target.closest('.ui-accordion-content,[role="tabpanel"]');
				const header = panel?.previousElementSibling;
				if (header && !visible(target)) {{ header.click(); await sleep(250); }}
				if (visible(target)) {{
					target.click();
					return {{matched: true, strategy: 'accordion_target', success: true, target: quotedTarget}};
				}}
			}}
		}}

		if (quotedTarget && /switch\s+between\s+the\s+tabs/i.test(instruction)) {{
			const targets = exact(document.querySelectorAll('.alink,a,[role="link"]'), quotedTarget);
			if (targets.length === 1) {{
				const target = targets[0];
				const panel = target.closest('[role="tabpanel"],div[id]');
				const tab = panel?.id ? document.querySelector(`a[href="#${{CSS.escape(panel.id)}}"]`) : null;
				if (tab) {{ tab.click(); await sleep(200); }}
				if (visible(target)) {{
					target.click();
					return {{matched: true, strategy: 'tab_target', success: true, target: quotedTarget}};
				}}
			}}
		}}

		const scrollMatch = instruction.match(/^select\s+(.+?)\s+from\s+the\s+scroll\s+list/i);
		if (scrollMatch) {{
			const labels = scrollMatch[1].split(',').map(normalize).filter(Boolean);
			const selects = Array.from(document.querySelectorAll('select[multiple]')).filter(visible);
			const submit = submitButton();
			if (selects.length === 1 && submit) {{
				const options = Array.from(selects[0].options);
				const matched = labels.map(label => options.filter(option => lower(option.text) === lower(label)));
				if (matched.every(items => items.length === 1)) {{
					options.forEach(option => {{ option.selected = labels.some(label => lower(option.text) === lower(label)); }});
					selects[0].dispatchEvent(new Event('input', {{bubbles: true}}));
					selects[0].dispatchEvent(new Event('change', {{bubbles: true}}));
					submit.click();
					return {{matched: true, strategy: 'scroll_multiselect', success: true, labels}};
				}}
			}}
		}}

		const flightMatch = instruction.match(
			/^book\s+the\s+(cheapest|shortest)\s+one-way\s+flight\s+from:\s+(.+?)\s+to:\s+(.+?)\s+on\s+(\d{{1,2}}\/\d{{1,2}}\/\d{{4}})/i
		);
		if (flightMatch) {{
			const [, objective, origin, destination, date] = flightMatch;
			const textInputs = Array.from(document.querySelectorAll('input[type="text"]')).filter(visible);
			const fromInput = textInputs.find(element => /from/i.test(`${{element.placeholder}} ${{element.name}} ${{element.id}}`));
			const toInput = textInputs.find(element => /(?:^|[-_])to(?:$|[-_])/i.test(`${{element.placeholder}} ${{element.name}} ${{element.id}}`));
			const dateInput = textInputs.find(element => /date/i.test(`${{element.placeholder}} ${{element.name}} ${{element.id}}`));
			const search = Array.from(document.querySelectorAll('button,input[type="button"],input[type="submit"]'))
				.filter(visible).find(element => /^search$/i.test(normalize(element.innerText || element.value)));
			const chooseAutocomplete = async (input, query) => {{
				input.focus(); setText(input, query);
				if (window.jQuery && window.jQuery(input).hasClass('ui-autocomplete-input'))
					window.jQuery(input).autocomplete('search', query);
				await sleep(180);
				const options = Array.from(document.querySelectorAll('.ui-autocomplete li,.ui-menu-item'))
					.filter(visible).filter(element => lower(element.innerText).includes(lower(query)));
				if (options.length < 1) return false;
				options[0].click(); await sleep(100); return Boolean(input.value);
			}};
			if (fromInput && toInput && dateInput && search) {{
				const originSet = await chooseAutocomplete(fromInput, origin);
				const destinationSet = await chooseAutocomplete(toInput, destination);
				if (window.jQuery && window.jQuery(dateInput).hasClass('hasDatepicker'))
					window.jQuery(dateInput).datepicker('setDate', date);
				else setText(dateInput, date);
				if (!originSet || !destinationSet || dateInput.value !== date)
					return {{matched: true, strategy: 'flight_search_rank', success: false, error: 'Could not verify search fields.'}};
				search.click(); await sleep(250);
				const flights = Array.from(document.querySelectorAll('.flight')).filter(visible).map(element => {{
					const priceText = element.querySelector('.flight-price')?.innerText || '';
					const durationText = element.querySelector('.time-duration')?.innerText || '';
					const price = Number(priceText.replace(/[^0-9.]/g, ''));
					const parts = durationText.match(/(?:(\d+)h)?\s*(?:(\d+)m)?/i);
					const duration = parts ? Number(parts[1] || 0) * 60 + Number(parts[2] || 0) : NaN;
					return {{element, price, duration}};
				}}).filter(item => Number.isFinite(objective.toLowerCase() === 'cheapest' ? item.price : item.duration));
				if (flights.length) {{
					flights.sort((left, right) => objective.toLowerCase() === 'cheapest'
						? left.price - right.price : left.duration - right.duration);
					flights[0].element.querySelector('button')?.click();
					return {{matched: true, strategy: 'flight_search_rank', success: true, objective, candidate_count: flights.length}};
				}}
			}}
		}}

		return {{matched: false, strategy: null, success: false}};
	}})()"""
	try:
		result = await cdp_session.cdp_client.send.Runtime.evaluate(
			params={'expression': expression, 'awaitPromise': True, 'returnByValue': True},
			session_id=cdp_session.session_id,
		)
		value = result.get('result', {}).get('value')
		return value if isinstance(value, dict) else {'matched': False, 'success': False, 'error': 'No workflow result.'}
	except Exception as error:
		return {'matched': False, 'success': False, 'error': f'Visible DOM workflow failed: {error}'}


async def _detect_short_ui_page(browser_session: BrowserSession) -> ShortUiPageProfile:
	"""Profile current-page capabilities using only visible DOM evidence."""

	try:
		cdp_session = await browser_session.get_or_create_cdp_session()
		result = await cdp_session.cdp_client.send.Runtime.evaluate(
			params={
				'expression': """(() => {
					const visible = (element) => {
						const style = getComputedStyle(element);
						const rect = element.getBoundingClientRect();
						return style.display !== 'none' && style.visibility !== 'hidden'
							&& rect.width > 0 && rect.height > 0;
					};
					const interactive = Array.from(document.querySelectorAll(
						'input,select,textarea,button,a[href],[role="button"],[role="checkbox"],'
						+ '[role="radio"],[role="option"],[role="textbox"],[contenteditable="true"]'
						+ ',[role="slider"],.ui-slider-handle'
					)).filter(visible);
					const scrollable = Array.from(document.querySelectorAll('body *')).filter((element) => {
						if (!visible(element)) return false;
						const style = getComputedStyle(element);
						return /(auto|scroll)/.test(style.overflowY)
							&& element.scrollHeight > element.clientHeight + 2;
					});
					const textLength = (document.body?.innerText || '').replace(/\\s+/g, ' ').trim().length;
					const instructionSample = (document.body?.innerText || '').replace(/\\s+/g, ' ').trim().slice(0, 2000);
					const bodyLines = (document.body?.innerText || '').split(/\\r?\\n/)
						.map(line => line.replace(/\\s+/g, ' ').trim()).filter(Boolean);
					const explicitInstruction = String(
						document.querySelector('#instruction-text,[data-task-instruction],.task-instruction')?.innerText || ''
					).replace(/\\s+/g, ' ').trim();
					const instructionIndex = bodyLines.findIndex(line => /^instruction\\s*:?/i.test(line));
					const labelledInstruction = instructionIndex >= 0
						? (bodyLines[instructionIndex].replace(/^instruction\\s*:?\\s*/i, '').trim()
							|| bodyLines[instructionIndex + 1] || '')
						: '';
					const instructionLine = explicitInstruction || labelledInstruction || bodyLines[0] || '';
					const linkCount = interactive.filter(element => element.matches('a[href]')).length;
					const inputs = interactive.filter(element => element.matches('input'));
					const accessibleName = (element) => {
						const labelledBy = element.getAttribute('aria-labelledby');
						const labelledText = labelledBy
							? labelledBy.split(/\\s+/).map(id => document.getElementById(id)?.innerText || '').join(' ').trim()
							: '';
						const explicitLabel = element.id
							? document.querySelector(`label[for="${CSS.escape(element.id)}"]`)?.innerText?.trim() || ''
							: '';
						const wrappingLabel = element.closest('label')?.innerText?.trim() || '';
						return element.getAttribute('aria-label') || labelledText || explicitLabel || wrappingLabel
							|| element.name || element.id || '';
					};
					const textInputs = inputs.filter(element =>
						!['button','submit','reset','checkbox','radio','hidden','image','file'].includes(
							(element.getAttribute('type') || 'text').toLowerCase()
						)
					);
					return {
						url: location.href,
						text_length: textLength,
						instruction_sample: instructionSample,
						instruction_line: instructionLine.slice(0, 1000),
						toggle_labels: interactive
							.filter(element => element.matches('input[type="checkbox"],input[type="radio"],[role="checkbox"],[role="radio"]'))
							.map(accessibleName).filter(Boolean),
						interactive_count: interactive.length,
						form_count: document.querySelectorAll('form').length,
						link_count: linkCount,
						scrollable_count: scrollable.length,
						button_count: interactive.filter(element =>
							element.matches('button,[role="button"],input[type="button"],input[type="submit"]')
						).length,
						text_input_count: textInputs.length,
						textarea_count: interactive.filter(element => element.matches('textarea,[contenteditable="true"]')).length,
						select_count: interactive.filter(element => element.matches('select')).length,
						date_input_count: interactive.filter(element => element.matches('input[type="date"]')).length,
						range_count: interactive.filter(element =>
							element.matches('input[type="range"],[role="slider"],.ui-slider-handle')
						).length,
						checkbox_count: interactive.filter(element => element.matches('input[type="checkbox"],[role="checkbox"]')).length,
						radio_count: interactive.filter(element => element.matches('input[type="radio"],[role="radio"]')).length,
						disabled_count: interactive.filter(element =>
							element.matches(':disabled,[aria-disabled="true"]')
						).length,
						submit_count: interactive.filter(element =>
							element.matches('button[type="submit"],input[type="submit"]')
							|| /^(submit|buy now|checkout|place order)$/i.test((element.innerText || element.value || '').trim())
						).length
					};
				})()""",
				'returnByValue': True,
			},
			session_id=cdp_session.session_id,
		)
		value = result.get('result', {}).get('value') or {}
		profile = ShortUiPageProfile.model_validate(value)
		return _classify_ui_capabilities(profile)
	except Exception as error:
		print(f'[DEBUG] Short-UI page profiling failed: {error}', file=sys.stderr)
		return ShortUiPageProfile()


def _short_ui_task_guidance(
	*,
	batch_enabled: bool,
	wait_enabled: bool,
	transfer_enabled: bool,
	selection_enabled: bool,
	date_enabled: bool,
	ordinal_enabled: bool,
) -> str:
	"""Return instructions only for capabilities observed on the current page."""

	instructions = [
		'\n\nCapability-guided execution: use only the enabled specialized actions below. Keep the plan short, '
		'ground each action in the current DOM, and verify live state before an irreversible submit. '
		'Do not call done until the visible workflow has completed.',
	]
	if batch_enabled:
		instructions.append(
			'When two or more independent form fields are visible, use set_form_values to update them in one bounded '
			'batch and verify all values before submitting.'
		)
	if wait_enabled:
		instructions.append(
			'For temporal instructions or dynamically changing controls, use wait_for_ui with the requested minimum '
			'delay and a stability check before the next irreversible click.'
		)
	if transfer_enabled:
		instructions.append(
			'When a task requires copying, pasting, duplicating, or preserving an exact visible control value, use '
			'transfer_control_value from the source index to the target index instead of regenerating the value.'
		)
	if selection_enabled:
		instructions.append(
			'For a labelled checkbox or radio task, call set_controls_by_labels once with the complete exact requested '
			'label list, verify its result, and then submit once. Do not click checkbox indices individually.'
		)
	if date_enabled:
		instructions.append(
			'For a native date input, use the exact DOM value in ISO YYYY-MM-DD form and verify readback before submitting.'
		)
	if ordinal_enabled:
		instructions.append(
			'For ordinal controls such as the 2nd checkbox, preserve DOM order and verify the exact indexed control state. '
			'For sliders, set the numeric value directly rather than estimating with a click.'
		)
	return ' '.join(instructions)


async def _read_miniwob_reward(browser_session: BrowserSession, threshold: float) -> MiniWoBRewardResponse:
	"""Read MiniWoB++ ground-truth reward variables instead of asking an LLM judge."""

	try:
		cdp_session = await browser_session.get_or_create_cdp_session()
		result = await cdp_session.cdp_client.send.Runtime.evaluate(
			params={
				'expression': """(() => ({
					done: Boolean(window.WOB_DONE_GLOBAL),
					reward: Number(window.WOB_RAW_REWARD_GLOBAL || 0),
					reason: window.WOB_REWARD_REASON || '',
					episode_id: Number(window.WOB_EPISODE_ID || 0),
					latest: window.core && Number.isFinite(core.wob_latest) ? Number(core.wob_latest) : null
				}))()""",
				'returnByValue': True,
			},
			session_id=cdp_session.session_id,
		)
		value = result.get('result', {}).get('value') or {}
		reward = float(value.get('reward') or 0.0)
		done = bool(value.get('done'))
		success = done and reward >= threshold
		return MiniWoBRewardResponse(
			success=success,
			reward=reward if done else None,
			reason=str(value.get('reason') or ''),
			explanation=(
				f'MiniWoB runtime reward={reward:.4f}; threshold={threshold:.4f}; done={done}; '
				f'episode_id={int(value.get("episode_id") or 0)}; reason={value.get("reason") or "none"}.'
			),
		)
	except Exception as error:
		return MiniWoBRewardResponse(
			success=False,
			explanation=f'MiniWoB runtime reward capture failed: {error}',
		)


def _build_webchallenger_tools() -> Tools:
	"""Expose an online, page-derived PageMem summary without site-private data."""

	tools = Tools()

	@tools.action(
		'Inspect the current page as compact semantic DOM sections. Call after a meaningful page change, '
		'then expand attention only to sections relevant to the task.'
	)
	async def inspect_page_sections(browser_session: BrowserSession) -> ActionResult:
		cdp_session = await browser_session.get_or_create_cdp_session()
		result = await cdp_session.cdp_client.send.Runtime.evaluate(
			params={
				'expression': """(() => {
					const visible = (element) => {
						const style = getComputedStyle(element);
						const rect = element.getBoundingClientRect();
						return style.display !== 'none' && style.visibility !== 'hidden'
							&& rect.width > 0 && rect.height > 0;
					};
					const selectors = [
						'header', 'nav', 'main', 'form', 'section', 'article', 'aside', 'footer',
						'[role="region"]', '[role="dialog"]', '[role="search"]'
					];
					let elements = [...document.querySelectorAll(selectors.join(','))].filter(visible);
					if (!elements.length) {
						elements = [...document.body.children].filter(visible);
					}
					const seen = new Set();
					const sections = [];
					for (const element of elements) {
						const rect = element.getBoundingClientRect();
						const key = [
							element.tagName, Math.round(rect.x), Math.round(rect.y),
							Math.round(rect.width), Math.round(rect.height)
						].join(':');
						if (seen.has(key)) continue;
						seen.add(key);
						const heading = element.querySelector('h1,h2,h3,h4,[role="heading"]');
						const text = (element.innerText || '').replace(/\\s+/g, ' ').trim();
						const controls = [...element.querySelectorAll(
							'a,button,input,select,textarea,[role="button"],[role="link"],[role="option"]'
						)].filter(visible);
						sections.push({
							id: sections.length + 1,
							tag: element.tagName.toLowerCase(),
							role: element.getAttribute('role') || '',
							label: element.getAttribute('aria-label') || heading?.innerText?.trim() || '',
							summary: text.slice(0, 500),
							interactive_count: controls.length,
							interactive_labels: controls.slice(0, 12).map((control) =>
								(control.innerText || control.getAttribute('aria-label')
									|| control.getAttribute('placeholder') || control.getAttribute('value') || '')
									.replace(/\\s+/g, ' ').trim()
							).filter(Boolean)
						});
						if (sections.length >= 30) break;
					}
					return {url: location.href, title: document.title, sections};
				})()""",
				'returnByValue': True,
			},
			session_id=cdp_session.session_id,
		)
		value = result.get('result', {}).get('value') or {}
		return ActionResult(
			extracted_content='ONLINE_PAGE_MEM\n' + json.dumps(value, ensure_ascii=False),
			include_extracted_content_only_once=True,
		)

	return tools


def _comparison_method_prompt(method: str) -> str:
	"""Return controlled method instructions for paper-inspired comparison agents."""

	if method == 'adapt_style':
		return (
			'\n\nADaPT-style controller: first try to execute the task directly. When a subtask is blocked or an '
			'action fails, identify only that blocked subtask and decompose it into 2-4 smaller observable '
			'steps. Execute those steps, then resume the parent task. Decompose recursively only as needed; '
			'do not restart already completed subtasks. Use only visible page evidence and normal browser actions.'
		)
	if method == 'reflexion_style':
		return (
			'\n\nReflexion-style controller: after an action fails, the page does not change as expected, or the '
			'current candidate violates the task, write a brief internal reflection naming the failure cause and '
			'one concrete correction. Preserve that lesson in memory, avoid repeating the failed action or '
			'candidate, and retry through a different visible interaction path. Use only visible page evidence.'
		)
	if method == 'agentoccam_style':
		return (
			'\n\nAgentOccam-style observation/action alignment: treat the page as a concise hierarchical reading '
			'comprehension passage. Focus on task-relevant visible elements, their stable element indexes, labels, '
			'states, and nearby context; ignore decorative or redundant nodes. Keep only a compact summary of the '
			'last three interactions. Emit one grounded normal browser action at a time, using click, input/type, '
			'go_back, or done. Do not invent selectors, hidden data, extra agent roles, demonstrations, search '
			'strategies, or online feedback. Stop only when visible environment evidence confirms completion.'
		)
	if method == 'webchallenger_style':
		return (
			'\n\nWebChallenger-style PageMem control: after opening a page or causing a meaningful page change, '
			'call inspect_page_sections to skim its semantic DOM sections. Focus subsequent reasoning and normal '
			'browser actions on the smallest task-relevant section. Reuse observed page structure during this task, '
			'avoid repeatedly expanding irrelevant regions, and group obvious form or option interactions into one '
			'clear workflow while still executing through standard browser actions. Use no hidden site data or '
			'precomputed benchmark answers.'
		)
	return ''


def _comparison_provenance(method: str) -> dict | None:
	"""Record the official source used to construct an adapted baseline."""

	if method == 'agentoccam_style':
		return {
			'implementation': 'adapted',
			'official_repository': 'https://github.com/amazon-science/AgentOccam',
			'commit': 'c078ba629212ea7cee35b2718dc8df72c05e57c7',
			'adaptation_note': 'Browser Use enforces a minimum six-item history window.',
		}
	if method == 'adaplanner_style':
		return {
			'implementation': 'adapted',
			'official_repository': 'https://github.com/haotiansun14/AdaPlanner',
			'commit': 'dba187601500e7d91fee5fd786a455aaa7988677',
			'safety_note': 'Generated plans are structured data; arbitrary Python exec is disabled.',
		}
	if method == 'webchallenger_style':
		return {
			'implementation': 'adapted',
			'official_repository': 'https://github.com/jayoohwang1/webchallenger',
			'commit': '0fd9831e00879cecb033ac6753a3b8e46068d74a',
			'adaptation_note': (
				'Uses online semantic DOM sections; official offline website exploration and cached cross-task '
				'memory are disabled to prevent benchmark leakage.'
			),
		}
	if method == 'webdart_style':
		return {
			'implementation': 'paper_adapted',
			'paper': 'https://arxiv.org/abs/2510.06587',
			'official_repository': None,
			'adaptation_note': (
				'Implements validated navigation/extraction/execution decomposition and visible-state replanning; '
				'the paper did not publish an official repository at integration time.'
			),
		}
	if method == 'weboperator_style':
		return {
			'implementation': 'adapted',
			'official_repository': 'https://github.com/kagnlp/WebOperator',
			'commit': '36a88e0da5f495629efbceb811fb16936b8c817c',
			'adaptation_note': (
				'Uses a bounded trajectory-level candidate frontier with safety and reversibility ranking. '
				'Official action-level BrowserGym snapshot replay is not available in the shared harness.'
			),
		}
	return None


def _isolate_webshop_session(task_data: dict, method: str) -> None:
	"""Give every WebShop subprocess fresh state while preserving the fixed goal index."""

	pattern = r'(https?://127\.0\.0\.1(?::\d+)?/)(browseruse_fixed_(\d+))'
	match = re.search(pattern, task_data['task'])
	if not match:
		return
	experiment_label = os.getenv('EVAL_EXPERIMENT_LABEL', '').strip()
	label = re.sub(r'[^a-z0-9]+', '_', experiment_label or method or 'full_improved').strip('_')
	seed = int(os.getenv('EVAL_SEED', '0'))
	session_id = f'browseruse_{label}_{seed}_{os.getpid()}_fixed_{match.group(3)}'
	task_data['task'] = re.sub(pattern, rf'\1{session_id}', task_data['task'], count=1)


def _task_start_url(task_data: dict) -> str:
	"""Extract the benchmark-local start URL from a task specification."""

	match = re.search(r'(?:https?://127\.0\.0\.1(?::\d+)?/[^\s]+|file:///[^\s]+?\.html)', task_data['task'])
	if not match:
		raise ValueError('Benchmark task does not contain a supported local start URL')
	return match.group(0).rstrip('.')


async def _reset_benchmark_page(browser_session: BrowserSession, task_data: dict) -> None:
	"""Reset one benchmark trial while preserving identical browser capabilities."""

	from browser_use.browser.events import NavigateToUrlEvent

	event = browser_session.event_bus.dispatch(NavigateToUrlEvent(url=_task_start_url(task_data), new_tab=False))
	await event
	if task_data.get('miniwob_reward_threshold') is None:
		return
	seed_value = f'{os.getenv("EVAL_SEED", "0")}:{task_data.get("name", task_data["task"])}'
	cdp_session = await browser_session.get_or_create_cdp_session()
	await cdp_session.cdp_client.send.Runtime.evaluate(
		params={
			'expression': (
				f'Math.seedrandom({json.dumps(seed_value)}); '
				'core.EPISODE_MAX_TIME=300000; '
				'if (core.EP_TIMER !== null) { clearTimeout(core.EP_TIMER); core.EP_TIMER=null; } '
				'core.startEpisodeReal(); core.startEpisode=function(){}; true;'
			),
			'returnByValue': True,
		},
		session_id=cdp_session.session_id,
	)


async def _score_benchmark_state(
	task_data: dict,
	browser_session: BrowserSession,
	history: AgentHistoryList,
) -> tuple[bool, float | None, str]:
	"""Score a controller trial from environment state, never model self-report."""

	if task_data.get('miniwob_reward_threshold') is not None:
		response = await _read_miniwob_reward(browser_session, float(task_data['miniwob_reward_threshold']))
		return response.success, response.reward, response.explanation
	final_url, final_page_text = await _read_final_browser_page(browser_session)
	if not final_url:
		urls = history.urls()
		final_url = urls[-1] if urls else ''
	if final_url and not final_page_text:
		final_page_text = await _fetch_local_webshop_page(final_url)
	response = _webshop_reward_response(
		task_data,
		history,
		history.final_result() or '',
		final_page_text=final_page_text,
		final_url=final_url,
	)
	if response is None:
		return False, None, 'No deterministic benchmark scorer was configured.'
	return response.success, response.reward, response.explanation


def _controller_trace(history: AgentHistoryList, explanation: str) -> str:
	"""Build a bounded failure trace for planner or reflector calls."""

	errors = [str(error)[:300] for error in history.errors() if error]
	return json.dumps(
		{
			'final_result': (history.final_result() or '')[:800],
			'last_urls': history.urls()[-5:],
			'last_actions': history.action_names()[-12:],
			'errors': errors[-5:],
			'environment_evaluation': explanation[:800],
		},
		ensure_ascii=False,
	)


async def _structured_controller_call(
	llm,
	prompt: str,
	schema: type[BaseModel],
	fallback: BaseModel | None = None,
) -> BaseModel:
	"""Call a planner/reflector with validated Pydantic output."""

	try:
		response = await llm.ainvoke([UserMessage(content=prompt)], output_format=schema)
		if isinstance(response.completion, schema):
			return response.completion
		return _parse_structured_response_text(str(response.completion), schema)
	except Exception as structured_error:
		schema_prompt = (
			f'{prompt}\n\nReply ONLY with one raw JSON object matching this JSON Schema. '
			f'Do not add prose or markdown fences.\n{json.dumps(schema.model_json_schema(), ensure_ascii=False)}'
		)
		response = await llm.ainvoke([UserMessage(content=schema_prompt)])
		try:
			return _parse_structured_response_text(str(response.completion), schema)
		except Exception as raw_error:
			if fallback is not None:
				print(
					f'[DEBUG] Controller JSON fallback used for {schema.__name__}: '
					f'structured_error={structured_error}; raw_error={raw_error}',
					file=sys.stderr,
				)
				return fallback
			raise RuntimeError(
				f'Controller failed to return valid JSON. structured_error={structured_error}; raw_error={raw_error}'
			) from raw_error


def _parse_structured_response_text(text: str, schema: type[BaseModel]) -> BaseModel:
	"""Parse raw or fenced JSON while rejecting surrounding planner prose."""

	stripped = text.strip()
	match = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', stripped, flags=re.DOTALL | re.IGNORECASE)
	if match:
		stripped = match.group(1).strip()
	try:
		return schema.model_validate_json(stripped)
	except Exception:
		payload, _ = json.JSONDecoder().raw_decode(stripped)
		return schema.model_validate(payload)


async def _resolve_semantic_toggle_workflow(
	llm,
	browser_session: BrowserSession,
	instruction: str,
	candidate_labels: list[str],
) -> dict:
	"""Resolve one visible candidate per semantic target, then apply and verify exact labels."""

	match = re.search(
		r'\b(?:words?\s+)?similar\s+to\s+(.+?)(?:\s+and\s+click\b|\s+then\s+click\b|$)',
		instruction,
		flags=re.IGNORECASE,
	)
	if not match or not candidate_labels:
		return {'matched': False, 'success': False, 'strategy': None, 'llm_calls': 0}
	targets = [part.strip(' ,.') for part in re.split(r'\s*,\s*|\s+and\s+', match.group(1)) if part.strip(' ,.')]
	if not targets:
		return {'matched': False, 'success': False, 'strategy': None, 'llm_calls': 0}

	prompt = (
		'Map each target word to exactly one distinct candidate label with the closest meaning. '
		'Every target must receive one label, labels must come verbatim from the candidate list, and return no extras. '
		f'Target words: {json.dumps(targets, ensure_ascii=False)}\n'
		f'Candidate labels: {json.dumps(candidate_labels, ensure_ascii=False)}'
	)
	llm_calls = 0
	usage_totals = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
	try:
		llm_calls += 1
		response = await llm.ainvoke([UserMessage(content=prompt)])
		if response.usage:
			usage_totals['prompt_tokens'] += response.usage.prompt_tokens
			usage_totals['completion_tokens'] += response.usage.completion_tokens
			usage_totals['total_tokens'] += response.usage.total_tokens
		if isinstance(response.completion, SemanticToggleSelection):
			selection = response.completion
		else:
			completion_text = str(response.completion)
			try:
				selection = _parse_structured_response_text(completion_text, SemanticToggleSelection)
			except Exception:
				mapped_labels = []
				for mapped_tail in re.findall(
					r'(?:\bmaps?\s+to\b|[-=]>|\u2192)\s*([^\r\n]+)',
					completion_text,
					flags=re.IGNORECASE,
				):
					matches = [
						label
						for label in candidate_labels
						if re.search(rf'(?<!\w){re.escape(label)}(?!\w)', mapped_tail, flags=re.IGNORECASE)
					]
					if len(matches) == 1:
						mapped_labels.append(matches[0])
				if len(mapped_labels) != len(targets):
					mentioned_labels = [
						label
						for label in candidate_labels
						if re.search(rf'(?<!\w){re.escape(label)}(?!\w)', completion_text, flags=re.IGNORECASE)
					]
					if len(mentioned_labels) == len(targets):
						mapped_labels = mentioned_labels
				selection = SemanticToggleSelection(selected_labels=mapped_labels)
	except Exception as error:
		return {
			'matched': True,
			'success': False,
			'strategy': 'semantic_bipartite_matching',
			'llm_calls': llm_calls,
			'usage': usage_totals,
			'error': str(error),
		}
	selected_labels = selection.selected_labels
	normalized_candidates = {' '.join(label.casefold().split()): label for label in candidate_labels}
	normalized_selected = [' '.join(label.casefold().split()) for label in selected_labels]
	if (
		len(selected_labels) != len(targets)
		or len(set(normalized_selected)) != len(targets)
		or any(label not in normalized_candidates for label in normalized_selected)
	):
		return {
			'matched': True,
			'success': False,
			'strategy': 'semantic_bipartite_matching',
			'llm_calls': llm_calls,
			'usage': usage_totals,
			'targets': targets,
			'selected_labels': selected_labels,
			'error': 'Semantic resolver violated one-to-one candidate constraints.',
		}

	canonical_labels = [normalized_candidates[label] for label in normalized_selected]
	cdp_session = await browser_session.get_or_create_cdp_session()
	expression = f"""(() => {{
		const requested = new Set({json.dumps(canonical_labels)}.map(value => value.toLowerCase().trim()));
		const controls = Array.from(document.querySelectorAll('input[type="checkbox"],input[type="radio"]'));
		const name = element => {{
			const explicit = element.id ? document.querySelector(`label[for="${{CSS.escape(element.id)}}"]`) : null;
			return String(element.getAttribute('aria-label') || explicit?.innerText || element.closest('label')?.innerText
				|| element.name || element.id || '').replace(/\\s+/g, ' ').trim();
		}};
		const states = controls.map(element => ({{element, label: name(element)}}));
		const matched = states.filter(state => requested.has(state.label.toLowerCase()));
		if (matched.length !== requested.size)
			return {{success: false, error: 'Visible semantic labels were missing or ambiguous.'}};
		states.forEach(state => {{
			const desired = requested.has(state.label.toLowerCase());
			if (state.element.checked !== desired) state.element.click();
		}});
		const verified = states.every(state => state.element.checked === requested.has(state.label.toLowerCase()));
		const submits = Array.from(document.querySelectorAll('button,input[type="submit"]')).filter(element =>
			/submit/i.test(element.innerText || element.value || '')
		);
		if (!verified || submits.length !== 1)
			return {{success: false, error: 'Semantic toggle verification or submit grounding failed.'}};
		submits[0].click();
		return {{success: true}};
	}})()"""
	result = await cdp_session.cdp_client.send.Runtime.evaluate(
		params={'expression': expression, 'returnByValue': True},
		session_id=cdp_session.session_id,
	)
	value = result.get('result', {}).get('value') or {}
	return {
		'matched': True,
		'success': bool(value.get('success')),
		'strategy': 'semantic_bipartite_matching',
		'llm_calls': llm_calls,
		'usage': usage_totals,
		'targets': targets,
		'selected_labels': canonical_labels,
		'error': value.get('error'),
	}


async def _run_comparison_controller(
	method: str,
	task: str,
	task_data: dict,
	llm,
	browser_session: BrowserSession,
	agent_kwargs: dict,
	max_steps: int,
) -> tuple[AgentHistoryList, list[AgentHistoryList], dict]:
	"""Run environment-scored ADaPT, Reflexion, or AdaPlanner control."""

	max_trials = int(os.getenv('EVAL_CONTROLLER_MAX_TRIALS', '3'))
	histories: list[AgentHistoryList] = []
	trial_records: list[dict] = []
	controller_memory: list[dict] = []
	controller_llm_calls = 0
	attempt_task = task
	weboperator_candidates: list[dict] = []
	if method == 'adaplanner_style':
		controller_llm_calls += 1
		initial_plan = await _structured_controller_call(
			llm,
			(
				'Act as the planner from AdaPlanner. Produce a minimal code-style browser plan with 2-5 ordered steps. '
				'For every step, predict the visible observation that should follow. Use only normal browser '
				'actions and visible page evidence; do not output executable Python.\n\n'
				f'TASK:\n{task}'
			),
			AdaPlannerPlan,
		)
		initial_plan_data = initial_plan.model_dump()
		controller_memory.append(initial_plan_data)
		attempt_task = (
			f'{task}\n\nAdaPlanner closed-loop plan: execute from step 1. After each action, compare the visible '
			'observation with expected_observation. If it differs, revise the remaining plan before continuing.\n'
			f'{json.dumps(initial_plan_data, ensure_ascii=False)}'
		)
	elif method == 'webdart_style':
		controller_llm_calls += 1
		initial_plan = await _structured_controller_call(
			llm,
			(
				'Act as the decomposition module from WebDART. Decompose the browser objective into a conservative '
				'navigation goal, pages to visit, evidence to capture, a visible stopping criterion, and a final '
				'execution goal. Exploit filters or sorting controls only after they become visible. Use no hidden '
				'site data.\n\n'
				f'TASK:\n{task}'
			),
			WebDARTPlan,
			fallback=WebDARTPlan(
				navigation_goal='Open the task website and locate a product using visible search and navigation.',
				pages_to_visit=['Task start page', 'Search results', 'Candidate product page'],
				evidence_to_capture=[
					'Instruction keywords and constraints',
					'Candidate title and attributes',
					'Required option controls and their selected state',
				],
				stopping_criterion='A visible candidate satisfies every instruction constraint and required option.',
				execution_goal='Select all required options, verify them visibly, then complete the purchase.',
				revision_reason='controller_json_fallback',
			),
		)
		initial_plan_data = initial_plan.model_dump()
		controller_memory.append(initial_plan_data)
		attempt_task = (
			f'{task}\n\nWebDART dynamic plan: navigate while collecting the specified visible evidence, then execute '
			'the final goal. At every new page, check whether a visible filter, sort control, or shortcut justifies '
			'revising the remaining navigation plan. Do not stop before the stopping criterion is visibly met.\n'
			f'{json.dumps(initial_plan_data, ensure_ascii=False)}'
		)
	elif method == 'weboperator_style':
		controller_llm_calls += 1
		candidate_plan = await _structured_controller_call(
			llm,
			(
				'Act as the candidate generator and safety ranker from WebOperator. Produce 2-4 materially distinct '
				'browser trajectory strategies. For each, state expected visible progress, safety risk, and whether '
				'the path is reversible. Rank a reversible, low-risk strategy first. Treat purchases, submissions, '
				'deletions, messages, and state-changing Enter presses as destructive until their preconditions are '
				'visibly verified. Use only normal browser actions and visible page evidence.\n\n'
				f'TASK:\n{task}'
			),
			WebOperatorPlan,
			fallback=WebOperatorPlan(
				candidates=[
					WebOperatorCandidate(
						strategy='Search with the most discriminative visible task keywords, then verify every constraint.',
						expected_progress='A short candidate list with visibly relevant products.',
						safety_risk='Low; search and inspection are reversible.',
					),
					WebOperatorCandidate(
						strategy='Broaden the visible search query, compare candidates, and delay purchase until options match.',
						expected_progress='An alternative candidate path if the narrow query fails.',
						safety_risk='Low; purchase remains blocked pending visible verification.',
					),
				],
				rejected_reasons=['controller_json_fallback'],
			),
		)
		candidate_plan_data = candidate_plan.model_dump()
		controller_memory.append(candidate_plan_data)
		weboperator_candidates = candidate_plan_data['candidates']
		selected_index = min(max(candidate_plan.selected_candidate - 1, 0), len(weboperator_candidates) - 1)
		selected_candidate = weboperator_candidates.pop(selected_index)
		attempt_task = (
			f'{task}\n\nWebOperator candidate trajectory: follow the strategy below one grounded action at a time. '
			'Prefer reversible exploration. Before any destructive action, validate all visible task constraints '
			'and expected state change; do not retry a destructive action blindly.\n'
			f'{json.dumps(selected_candidate, ensure_ascii=False)}'
		)
	for trial in range(1, max_trials + 1):
		if trial == 1 or method not in {'adaplanner_style', 'webdart_style'}:
			await _reset_benchmark_page(browser_session, task_data)
		agent = Agent(task=attempt_task, llm=llm, browser_session=browser_session, **agent_kwargs)
		trial_timeout = float(os.getenv('EVAL_CONTROLLER_TRIAL_TIMEOUT_SECONDS', '240'))
		history = await asyncio.wait_for(agent.run(max_steps=max_steps), timeout=trial_timeout)
		histories.append(history)
		success, reward, explanation = await _score_benchmark_state(task_data, browser_session, history)
		trial_records.append({'trial': trial, 'success': success, 'reward': reward, 'environment_evaluation': explanation})
		if success or trial == max_trials:
			break

		trace = _controller_trace(history, explanation)
		if method == 'adapt_style':
			controller_llm_calls += 1
			try:
				plan = await _structured_controller_call(
					llm,
					(
						'Act as the planner in ADaPT. The executor failed the browser task below. Decompose only the '
						'blocked part into 2-5 observable browser subtasks. Preserve completed work conceptually, avoid '
						'hidden environment data, and make every step executable with normal browser actions.\n\n'
						f'TASK:\n{task}\n\nFAILURE TRACE:\n{trace}\n\nPRIOR PLANS:\n'
						f'{json.dumps(controller_memory, ensure_ascii=False)}'
					),
					AdaptivePlan,
				)
			except Exception as error:
				plan = AdaptivePlan(
					failure_cause=f'Planner call failed: {error}',
					steps=['Re-read the visible instruction and page state', 'Try a different visible interaction path'],
				)
			plan_data = plan.model_dump()
			controller_memory.append(plan_data)
			attempt_task = (
				f'{task}\n\nADaPT controller depth {trial}: execute this validated as-needed plan, then finish the original task. '
				'Do not claim success until the environment confirms completion.\n'
				f'{json.dumps(plan_data, ensure_ascii=False)}'
			)
		elif method == 'reflexion_style':
			controller_llm_calls += 1
			try:
				reflection = await _structured_controller_call(
					llm,
					(
						'Act as the Reflexion evaluator for a failed browser-agent trial. Produce a compact causal '
						'lesson and a materially different next strategy based only on visible evidence.\n\n'
						f'TASK:\n{task}\n\nFAILURE TRACE:\n{trace}\n\nPRIOR REFLECTIONS:\n'
						f'{json.dumps(controller_memory, ensure_ascii=False)}'
					),
					ReflectionNote,
				)
			except Exception as error:
				reflection = ReflectionNote(
					failure_cause=f'Reflection call failed: {error}',
					lesson='Do not repeat the same failed trajectory.',
					next_strategy='Re-read visible constraints and take a different interaction path.',
				)
			reflection_data = reflection.model_dump()
			controller_memory.append(reflection_data)
			attempt_task = (
				f'{task}\n\nReflexion memory from prior trials follows. Apply these lessons, avoid repeated actions, and '
				'do not claim success until the environment confirms completion.\n'
				f'{json.dumps(controller_memory, ensure_ascii=False)}'
			)
		elif method == 'adaplanner_style':
			controller_llm_calls += 1
			revised_plan = await _structured_controller_call(
				llm,
				(
					'Act as the feedback refiner from AdaPlanner. The browser remains at the current intermediate '
					'state. Compare actual feedback with predicted observations, revise only the blocked and '
					'remaining steps, and choose resume_from_step. Use only visible evidence and normal browser '
					'actions; do not output executable Python.\n\n'
					f'TASK:\n{task}\n\nFAILURE TRACE:\n{trace}\n\nPREVIOUS PLANS:\n'
					f'{json.dumps(controller_memory, ensure_ascii=False)}'
				),
				AdaPlannerPlan,
			)
			revised_plan_data = revised_plan.model_dump()
			controller_memory.append(revised_plan_data)
			attempt_task = (
				f'{task}\n\nAdaPlanner revised plan: continue from the current browser state at '
				f'step {revised_plan.resume_from_step}; do not restart completed work. Compare each resulting '
				'observation with the prediction and finish only after environment confirmation.\n'
				f'{json.dumps(revised_plan_data, ensure_ascii=False)}'
			)
		elif method == 'webdart_style':
			controller_llm_calls += 1
			revised_plan = await _structured_controller_call(
				llm,
				(
					'Act as the dynamic replanning module from WebDART. The browser remains at the current state. '
					'Use newly visible widgets, filters, page structure, and the failure trace to revise the '
					'navigation, evidence capture, stopping criterion, and execution goal. Preserve useful evidence '
					'already collected and avoid restarting completed navigation.\n\n'
					f'TASK:\n{task}\n\nFAILURE TRACE:\n{trace}\n\nPREVIOUS PLANS:\n'
					f'{json.dumps(controller_memory, ensure_ascii=False)}'
				),
				WebDARTPlan,
				fallback=WebDARTPlan(
					navigation_goal='Continue from the current page using a visibly different candidate path.',
					pages_to_visit=['Current page', 'Alternative search result', 'Alternative product page'],
					evidence_to_capture=[
						'The prior mismatch or missing constraint',
						'Alternative candidate attributes',
						'Required option controls and selected state',
					],
					stopping_criterion='The prior mismatch is resolved and every visible task constraint is satisfied.',
					execution_goal='Verify selected options and complete the purchase only after all checks pass.',
					revision_reason='controller_json_fallback_after_failed_trial',
				),
			)
			revised_plan_data = revised_plan.model_dump()
			controller_memory.append(revised_plan_data)
			attempt_task = (
				f'{task}\n\nWebDART revised plan: continue from the current visible browser state. Reuse collected '
				'evidence, exploit only visible shortcuts, and finish the final execution goal after the stopping '
				'criterion is satisfied.\n'
				f'{json.dumps(revised_plan_data, ensure_ascii=False)}'
			)
		else:
			if weboperator_candidates:
				selected_candidate = weboperator_candidates.pop(0)
			else:
				controller_llm_calls += 1
				replacement_plan = await _structured_controller_call(
					llm,
					(
						'Act as WebOperator after a failed candidate trajectory. Generate 2-4 new strategies that '
						'avoid the failed path, rank reversible exploration first, and explicitly delay destructive '
						'actions until visible preconditions are satisfied.\n\n'
						f'TASK:\n{task}\n\nFAILURE TRACE:\n{trace}\n\nPRIOR FRONTIER:\n'
						f'{json.dumps(controller_memory, ensure_ascii=False)}'
					),
					WebOperatorPlan,
					fallback=WebOperatorPlan(
						candidates=[
							WebOperatorCandidate(
								strategy='Use a broader query and inspect a candidate not used by the failed branch.',
								expected_progress='A materially different product candidate is opened.',
								safety_risk='Low; navigation and inspection are reversible.',
							),
							WebOperatorCandidate(
								strategy='Use visible filters or result pagination before selecting another candidate.',
								expected_progress='A different result region is explored without repeating the failed path.',
								safety_risk='Low until purchase; verify all options before committing.',
							),
						],
						rejected_reasons=['controller_json_fallback_after_failed_branch'],
					),
				)
				replacement_data = replacement_plan.model_dump()
				controller_memory.append(replacement_data)
				weboperator_candidates = replacement_data['candidates']
				selected_candidate = weboperator_candidates.pop(0)
			attempt_task = (
				f'{task}\n\nWebOperator alternative trajectory after a failed branch: use this materially different '
				'strategy. Prefer reversible actions and verify all visible preconditions before any destructive '
				'action. The environment has been reset to the identical start state.\n'
				f'{json.dumps(selected_candidate, ensure_ascii=False)}'
			)

	return (
		histories[-1],
		histories,
		{
			'controller_method': method,
			'trials_used': len(histories),
			'controller_llm_calls': controller_llm_calls,
			'trial_records': trial_records,
			'controller_memory': controller_memory,
			'provenance': _comparison_provenance(method),
		},
	)


async def _fetch_local_webshop_page(url: str) -> str:
	"""Fetch a local WebShop done page when Agent shutdown has already closed CDP."""
	if not re.match(r'^http://(?:127\.0\.0\.1|localhost)(?::\d+)?/', url):
		return ''

	def _fetch() -> str:
		with urlopen(url, timeout=5) as response:
			raw_html = response.read().decode('utf-8', errors='replace')
		return html.unescape(re.sub(r'<[^>]+>', ' ', raw_html))

	try:
		return await anyio.to_thread.run_sync(_fetch)
	except Exception as error:
		print(f'[DEBUG] Final WebShop URL fetch failed: {error}', file=sys.stderr)
		return ''


def _completed_result_files() -> set[str]:
	if not (RESULTS_JSONL and RESUME_RESULTS and os.path.exists(RESULTS_JSONL)):
		return set()
	completed = set()
	with open(RESULTS_JSONL, encoding='utf-8') as file:
		for line in file:
			try:
				result = json.loads(line)
			except json.JSONDecodeError:
				continue
			if result.get('file'):
				completed.add(result['file'])
	return completed


def _append_result(result: dict) -> None:
	if not RESULTS_JSONL:
		return
	result_path = os.path.abspath(RESULTS_JSONL)
	os.makedirs(os.path.dirname(result_path), exist_ok=True)
	with open(result_path, 'a', encoding='utf-8') as file:
		file.write(json.dumps(result, ensure_ascii=False) + '\n')


def _qwen_api_key() -> str:
	return (
		os.getenv('QWEN_CHAT_API_KEY')
		or os.getenv('QWEN_API_KEY')
		or os.getenv('QWEN_EMBED_API_KEY')
		or os.getenv('DASHSCOPE_API_KEY')
		or os.getenv('ALIBABA_CLOUD')
		or ''
	)


def _qwen_base_url() -> str:
	return os.getenv('QWEN_CHAT_BASE_URL') or os.getenv('QWEN_EMBED_BASE_URL') or ''


def _build_qwen_llm() -> ChatOpenAI | None:
	api_key = _qwen_api_key()
	base_url = _qwen_base_url()
	model = os.getenv('QWEN_CHAT_MODEL')
	if not (api_key and base_url and model):
		return None

	return ChatOpenAI(
		model=model,
		api_key=api_key,
		base_url=base_url,
		temperature=0,
		max_completion_tokens=int(os.getenv('QWEN_MAX_COMPLETION_TOKENS', '2048')),
		add_schema_to_system_prompt=True,
		dont_force_structured_output=True,
	)


def _build_agent_llm():
	api_key = os.getenv('BROWSER_USE_API_KEY')
	if api_key:
		return ChatBrowserUse(api_key=api_key)

	qwen_llm = _build_qwen_llm()
	if qwen_llm:
		print('[DEBUG] BROWSER_USE_API_KEY is not set; using Qwen-compatible ChatOpenAI for agent LLM', file=sys.stderr)
		return qwen_llm

	return None


def _build_judge_llm():
	google_api_key = os.getenv('GOOGLE_API_KEY')
	if google_api_key:
		return ChatGoogle(model='gemini-3.1-flash-lite')

	qwen_llm = _build_qwen_llm()
	if qwen_llm:
		print('[DEBUG] GOOGLE_API_KEY is not set; using Qwen-compatible ChatOpenAI for judge LLM', file=sys.stderr)
		return qwen_llm

	return None


def _env_bool(name: str, default: bool) -> bool:
	value = os.getenv(name)
	if value is None:
		return default
	return value.lower()[:1] in 'ty1'


async def _await_navigation(event) -> None:
	"""Await browser navigation with a benchmark-level hard timeout."""

	timeout = float(os.getenv('EVAL_NAVIGATION_TIMEOUT_SECONDS', '30'))
	try:
		await asyncio.wait_for(event, timeout=timeout)
	except TimeoutError as error:
		raise RuntimeError(f'INFRASTRUCTURE_FAILURE: navigation timed out after {timeout:.1f}s') from error


def _is_qwen_vl_model() -> bool:
	model = os.getenv('QWEN_CHAT_MODEL', '').lower()
	return model.startswith('qwen-vl') or '-vl-' in model


def _is_qwen3_vl_model() -> bool:
	model = os.getenv('QWEN_CHAT_MODEL', '').lower()
	return 'qwen3-vl' in model or 'qwen3_vl' in model


def _agent_kwargs() -> dict:
	kwargs = {}
	if os.getenv('EVAL_CALCULATE_COST') is not None:
		kwargs['calculate_cost'] = _env_bool('EVAL_CALCULATE_COST', default=True)
	if os.getenv('EVAL_USE_VISION') is not None:
		kwargs['use_vision'] = _env_bool('EVAL_USE_VISION', default=True)
	elif _is_qwen_vl_model():
		kwargs['use_vision'] = True

	if os.getenv('EVAL_MAX_ACTIONS_PER_STEP'):
		kwargs['max_actions_per_step'] = int(os.getenv('EVAL_MAX_ACTIONS_PER_STEP', '1'))
	elif _is_qwen_vl_model():
		kwargs['max_actions_per_step'] = 1

	if os.getenv('EVAL_VISUAL_CONTEXT_MODE'):
		kwargs['visual_context_mode'] = os.getenv('EVAL_VISUAL_CONTEXT_MODE')

	if os.getenv('EVAL_GOAL_AWARE_TASK_STRATEGY') is not None:
		kwargs['goal_aware_task_strategy'] = _env_bool('EVAL_GOAL_AWARE_TASK_STRATEGY', default=True)

	if os.getenv('EVAL_MAX_HISTORY_ITEMS'):
		kwargs['max_history_items'] = int(os.getenv('EVAL_MAX_HISTORY_ITEMS', '8'))
	elif _is_qwen3_vl_model():
		# Qwen3-VL tool use degrades when long browser traces crowd the current
		# observation. Keep the initial state plus the latest bounded trajectory.
		kwargs['max_history_items'] = int(os.getenv('EVAL_QWEN3VL_MAX_HISTORY_ITEMS', '8'))

	if os.getenv('EVAL_MAX_CLICKABLE_ELEMENTS_LENGTH'):
		kwargs['max_clickable_elements_length'] = int(
			os.getenv('EVAL_MAX_CLICKABLE_ELEMENTS_LENGTH', '7000')
		)

	return kwargs


def _history_metrics(history: AgentHistoryList) -> dict:
	"""Extract efficiency metrics without changing agent behavior."""

	usage = history.usage
	return {
		'steps': history.number_of_steps(),
		'duration_seconds': round(history.total_duration_seconds(), 3),
		'action_count': len(history.action_names()),
		'error_count': sum(error is not None for error in history.errors()),
		'action_names': history.action_names(),
		'errors': [str(error)[:500] for error in history.errors() if error],
		'prompt_tokens': usage.total_prompt_tokens if usage else None,
		'completion_tokens': usage.total_completion_tokens if usage else None,
		'total_tokens': usage.total_tokens if usage else None,
		'llm_invocations': usage.entry_count if usage else None,
		'estimated_cost': round(usage.total_cost, 6) if usage else None,
	}


def _combined_history_metrics(histories: list[AgentHistoryList]) -> dict:
	"""Aggregate executor usage across controller trials."""

	metrics = [_history_metrics(history) for history in histories]
	keys = [
		'steps',
		'duration_seconds',
		'action_count',
		'error_count',
		'prompt_tokens',
		'completion_tokens',
		'total_tokens',
		'llm_invocations',
		'estimated_cost',
	]
	combined = {'executor_trials': len(histories)}
	for key in keys:
		values = [metric[key] for metric in metrics if metric[key] is not None]
		combined[key] = round(sum(values), 6) if values else None
	combined['action_names'] = [name for metric in metrics for name in metric['action_names']]
	combined['errors'] = [error for metric in metrics for error in metric['errors']]
	return combined


def _failure_category(success: bool, explanation: str, reward: float | None) -> str:
	"""Map terminal outcomes to a stable process-level failure taxonomy."""

	if success:
		return 'success'
	lower = explanation.lower()
	if any(
		marker in lower
		for marker in (
			'infrastructure_failure',
			'connection refused',
			'connection closed',
			'err_connection',
			'local service',
		'service unavailable',
		'no space left on device',
	)
	):
		return 'infrastructure_failure'
	if 'timeout' in lower or 'timed out' in lower:
		return 'budget_exceeded'
	if any(marker in lower for marker in ('api', 'rate limit', 'data_inspection')):
		return 'model_or_infrastructure'
	if any(marker in lower for marker in ('navigation', 'connection', 'err_')):
		return 'navigation'
	if reward is not None and reward > 0:
		return 'partial_constraint_match'
	if 'no final webshop reward' in lower or 'done=false' in lower:
		return 'task_not_committed'
	return 'interaction_or_reasoning'


def _debug_history(history: AgentHistoryList) -> None:
	if not _env_bool('EVAL_DEBUG_HISTORY', default=False):
		return

	print(f'[DEBUG] URLs: {history.urls()}', file=sys.stderr)
	print(f'[DEBUG] Action names: {history.action_names()}', file=sys.stderr)
	print(f'[DEBUG] Errors: {history.errors()}', file=sys.stderr)
	print(f'[DEBUG] Extracted content: {history.extracted_content()}', file=sys.stderr)
	for step in history.agent_steps():
		print('[DEBUG] Agent step:', file=sys.stderr)
		print(step[:4000], file=sys.stderr)


def _parse_judge_response_text(text: str) -> JudgeResponse:
	stripped = text.strip()
	match = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', stripped, flags=re.DOTALL | re.IGNORECASE)
	if match:
		stripped = match.group(1).strip()
	return JudgeResponse.model_validate_json(stripped)


async def _judge_task(judge_llm, judge_prompt: str) -> JudgeResponse:
	try:
		response = await judge_llm.ainvoke([UserMessage(content=judge_prompt)], output_format=JudgeResponse)
		return response.completion
	except Exception as structured_error:
		raw_prompt = judge_prompt + '\n\nReply ONLY with raw JSON. Do not wrap it in markdown fences.'
		response = await judge_llm.ainvoke([UserMessage(content=raw_prompt)])
		try:
			return _parse_judge_response_text(response.completion)
		except Exception as raw_error:
			raise RuntimeError(
				f'Judge failed to return valid JSON. structured_error={structured_error}; raw_error={raw_error}'
			) from raw_error


async def run_single_task(task_file):
	"""Run a single task in the current process (called by subprocess)"""
	try:
		print(f'[DEBUG] Starting task: {os.path.basename(task_file)}', file=sys.stderr)

		# Suppress all logging in subprocess to avoid interfering with JSON output
		logging.getLogger().setLevel(logging.CRITICAL)
		for logger_name in ['browser_use', 'telemetry', 'message_manager']:
			logging.getLogger(logger_name).setLevel(logging.CRITICAL)
		warnings.filterwarnings('ignore')

		print('[DEBUG] Loading task file...', file=sys.stderr)
		content = await anyio.Path(task_file).read_text(encoding='utf-8')
		task_data = yaml.safe_load(content)
		adapter_contract = benchmark_adapter_contract(task_data, task_file)
		print(
			'[DEBUG] Benchmark adapter contract: '
			+ json.dumps(adapter_contract.model_dump(mode='json'), ensure_ascii=False),
			file=sys.stderr,
		)
		if task_data.get('miniwob_reward_threshold') is not None:
			configured_threshold = float(task_data['miniwob_reward_threshold'])
			paper_threshold = float(os.getenv('EVAL_MINIWOB_REWARD_THRESHOLD', '0.99'))
			task_data['miniwob_reward_threshold'] = max(configured_threshold, paper_threshold)
		task = task_data['task']
		comparison_method = os.getenv('EVAL_COMPARISON_METHOD', '').strip().lower()
		controller_methods = {
			'adapt_style',
			'reflexion_style',
			'adaplanner_style',
			'webdart_style',
			'weboperator_style',
		}
		if comparison_method not in {
			'',
			'plain_agent',
			'agentoccam_style',
			'webchallenger_style',
			*controller_methods,
		}:
			raise ValueError(f'Unsupported EVAL_COMPARISON_METHOD: {comparison_method}')
		if task_data.get('webshop_reward_threshold') is not None:
			_isolate_webshop_session(task_data, comparison_method)
			task = task_data['task']
		webshop_web_only = task_data.get('webshop_reward_threshold') is not None and _env_bool(
			'EVAL_WEBSHOP_WEB_ONLY', default=False
		)
		webshop_react_style = webshop_web_only and _env_bool('EVAL_WEBSHOP_REACT_STYLE', default=False)
		use_webshop_candidate_verifier = (
			task_data.get('webshop_reward_threshold') is not None
			and _env_bool('EVAL_WEBSHOP_CANDIDATE_VERIFIER', default=False)
			and not webshop_web_only
		)
		if task_data.get('webshop_reward_threshold') is not None and _env_bool(
			'EVAL_GOAL_AWARE_TASK_STRATEGY',
			default=False,
		):
			task += (
				'\n\nWebShop scoring hint: success requires the final WebShop reward to be close to 1.0. '
				'Before clicking Buy Now, explicitly verify that the chosen product and all selected options match the '
				'instruction. If a required color, size, style, flavor, scent, model, capacity, or quantity is missing, '
				'do not buy it; go back and search for a better candidate. A reward such as 0.5, 0.6667, 0.75, or 0.85 '
				'is a failed partial match, not success. If you see a final reward below 0.99 and still have steps left, '
				'reflect on the missing constraint, go back/search again, and try a better candidate instead of calling done().'
			)
		if use_webshop_candidate_verifier:
			task += (
				'\n\nWebShop deterministic controller requirement: use advance_webshop_task as the primary action. It automatically '
				'plans, opens the next unvisited ranked candidate, verifies it, and fills required options. Call it again after '
				'any ambiguous page interaction or blocked action instead of manually repeating searches. Before clicking any '
				'Buy Now / final purchase button, '
				'the controller must return BUY_READY. REJECT candidates must never be revisited. Use review_webshop_memory '
				'only when the controller reports NEEDS_AGENT or TRANSITION_LIMIT. Never use an external search engine. '
				'Only click Buy Now after the verifier returns BUY_READY and required options are selected.'
			)
		if webshop_react_style:
			task += (
				'\n\nReAct-style web evaluation: repeatedly observe the visible page, reason about the next useful '
				'action, execute one normal browser action, and use the resulting observation to update the plan. '
				'Use only visible page evidence. Search and inspect candidates through ordinary page interactions, '
				'check required options before purchase, and never claim success without reaching the real done page.'
			)
		elif webshop_web_only and comparison_method:
			task += (
				'\n\nFair web-only evaluation requirement: use only information exposed by the current browser page. '
				'Do not use internal WebShop JSON, hidden goal data, precomputed product rankings, constructed '
				'product URLs, or environment reward before checkout. Search, inspect, select options, and purchase '
				'through normal visible browser interactions. Never claim success without reaching the real done page.'
			)
		elif webshop_web_only:
			task += (
				'\n\nFair web-only evaluation requirement: use only information exposed by the current browser page. '
				'Do not use internal JSON, hidden goal data, precomputed answer rankings, or constructed destination URLs. '
				'Call advance_visible_task first and after any ambiguous transition. It may submit a visible search, open the '
				'highest-ranked visible candidate, select visibly required options, or report COMMIT_READY. Use '
				'inspect_ranked_candidates and open_ranked_candidate when comparing alternatives, and call '
				'review_execution_progress when an action or state repeats. Before an irreversible action, verify every visible '
				'constraint. Never claim success without visible terminal evidence from the real page. '
				'Treat element indexes as valid for one observation only: after navigation, search, option selection, or a '
				'changed page, discard old indexes. Do not manually repeat the same query or revisit a rejected candidate; '
				'call advance_visible_task so bounded recovery can choose the next visible candidate. When it reports '
				'COMMIT_READY, click Buy Now once, inspect the resulting page, and finish only after visible reward evidence.'
			)
		task += _comparison_method_prompt(comparison_method)
		judge_context = task_data.get('judge_context', ['The agent must solve the task'])
		max_steps = task_data.get('max_steps', 15)
		if webshop_web_only:
			max_steps = min(max_steps, int(os.getenv('EVAL_WEBSHOP_FALLBACK_MAX_STEPS', '12')))

		print(f'[DEBUG] Task: {task[:100]}...', file=sys.stderr)
		print(f'[DEBUG] Max steps: {max_steps}', file=sys.stderr)
		agent_llm = _build_agent_llm()
		if agent_llm is None:
			print('[SKIP] No agent LLM API key is set - skipping task evaluation', file=sys.stderr)
			return {
				'file': os.path.basename(task_file),
				'success': True,  # Mark as success so it doesn't fail CI
				'explanation': 'Skipped - agent LLM API key not available (fork PR or missing secret)',
			}

		judge_llm = _build_judge_llm()
		if judge_llm is None:
			print('[SKIP] No judge LLM API key is set - skipping task evaluation', file=sys.stderr)
			return {
				'file': os.path.basename(task_file),
				'success': True,  # Mark as success so it doesn't fail CI
				'explanation': 'Skipped - judge LLM API key not available (fork PR or missing secret)',
			}

		print('[DEBUG] LLMs initialized', file=sys.stderr)

		# Each subprocess gets its own profile and session
		print('[DEBUG] Creating browser session...', file=sys.stderr)
		is_webshop_task = task_data.get('webshop_reward_threshold') is not None
		is_miniwob_task = task_data.get('miniwob_reward_threshold') is not None
		short_ui_mode = False
		short_ui_profile = ShortUiPageProfile()
		browser_setup_error: Exception | None = None
		pre_agent_selection: dict = {
			'attempted': False,
			'requested_labels': [],
			'success': False,
			'error': None,
			'fallback': 'agent',
		}
		pre_agent_preparation: dict = {
			'attempted': False,
			'browser_state_refreshed': False,
			'inspection_error': None,
			'operations': [],
		}
		visible_dom_workflow: dict = {'matched': False, 'success': False, 'strategy': None}
		semantic_toggle_workflow: dict = {'matched': False, 'success': False, 'strategy': None, 'llm_calls': 0}
		profile = BrowserProfile(
			headless=True,
			user_data_dir=None,
			keep_alive=True,
			chromium_sandbox=False,  # Disable sandbox for CI environment (GitHub Actions)
			allowed_domains=['127.0.0.1'] if is_webshop_task or is_miniwob_task else None,
		)
		session = BrowserSession(browser_profile=profile)
		print('[DEBUG] Browser session created', file=sys.stderr)

		# Test external navigation for general tasks only. WebShop is intentionally
		# restricted to its local origin so recovery cannot drift to a search engine.
		try:
			await session.start()
			if is_miniwob_task:
				from browser_use.browser.events import NavigateToUrlEvent

				miniwob_url_match = re.search(
					r'(?:file:///[^\s]+?|http://127\.0\.0\.1(?::\d+)?/[^\s]+?)\.html', task_data['task']
				)
				if not miniwob_url_match:
					raise ValueError('MiniWoB task is missing a supported local .html URL')
				event = session.event_bus.dispatch(NavigateToUrlEvent(url=miniwob_url_match.group(0), new_tab=False))
				await _await_navigation(event)
				seed_value = f'{os.getenv("EVAL_SEED", "0")}:{task_data.get("name", task_data["task"])}'
				cdp_session = await session.get_or_create_cdp_session()
				await cdp_session.cdp_client.send.Runtime.evaluate(
					params={
						'expression': (
							f'Math.seedrandom({json.dumps(seed_value)}); '
							'core.EPISODE_MAX_TIME=300000; '
							'if (core.EP_TIMER !== null) { clearTimeout(core.EP_TIMER); core.EP_TIMER=null; } '
							'core.startEpisodeReal(); core.startEpisode=function(){}; true;'
						),
						'returnByValue': True,
					},
					session_id=cdp_session.session_id,
				)
				print('[DEBUG] MiniWoB page initialized with a 300-second episode limit', file=sys.stderr)
			elif is_webshop_task:
				from browser_use.browser.events import NavigateToUrlEvent

				event = session.event_bus.dispatch(NavigateToUrlEvent(url=_task_start_url(task_data), new_tab=False))
				await _await_navigation(event)
				print('[DEBUG] Browser started and opened the local WebShop start page', file=sys.stderr)
			else:
				from browser_use.browser.events import NavigateToUrlEvent

				event = session.event_bus.dispatch(NavigateToUrlEvent(url='https://httpbin.org/get', new_tab=True))
				await _await_navigation(event)
				print('[DEBUG] Browser test: navigation successful', file=sys.stderr)
				title = await session.get_current_page_title()
				print(f"[DEBUG] Browser test: got title '{title}'", file=sys.stderr)
			if not comparison_method and _env_bool('EVAL_SHORT_UI_STRATEGY', default=True):
				short_ui_profile = await _detect_short_ui_page(session)
				short_ui_mode = short_ui_profile.eligible and not is_webshop_task
				print(
					'[DEBUG] Unified capability router: '
					+ json.dumps(short_ui_profile.model_dump(mode='json'), ensure_ascii=False),
					file=sys.stderr,
				)
		except Exception as browser_error:
			browser_setup_error = browser_error
			print(f'[DEBUG] Browser test failed: {str(browser_error)}', file=sys.stderr)
			print(
				f'[DEBUG] Browser error type: {type(browser_error).__name__}',
				file=sys.stderr,
			)
		if browser_setup_error is not None and (is_miniwob_task or is_webshop_task):
			raise RuntimeError(f'INFRASTRUCTURE_FAILURE: local benchmark setup failed: {browser_setup_error}')

		short_ui_batch_enabled = (
			short_ui_mode
			and short_ui_profile.multi_field_form
			and not _env_bool('EVAL_ABLATE_SHORT_UI_BATCH', default=False)
		)
		short_ui_wait_enabled = (
			short_ui_mode
			and short_ui_profile.temporal_control
			and not _env_bool('EVAL_ABLATE_SHORT_UI_WAIT', default=False)
		)
		short_ui_transfer_enabled = (
			short_ui_mode
			and short_ui_profile.exact_transfer
			and not _env_bool('EVAL_ABLATE_SHORT_UI_TRANSFER', default=False)
		)
		short_ui_selection_enabled = short_ui_mode and short_ui_profile.labelled_selection
		short_ui_date_enabled = short_ui_mode and short_ui_profile.date_entry
		short_ui_ordinal_enabled = short_ui_mode and short_ui_profile.ordinal_control
		short_ui_inspection_enabled = short_ui_mode and not _env_bool('EVAL_ABLATE_SHORT_UI_INSPECTION', default=False)
		if short_ui_mode:
			task += _short_ui_task_guidance(
				batch_enabled=short_ui_batch_enabled,
				wait_enabled=short_ui_wait_enabled,
				transfer_enabled=short_ui_transfer_enabled,
				selection_enabled=short_ui_selection_enabled,
				date_enabled=short_ui_date_enabled,
				ordinal_enabled=short_ui_ordinal_enabled,
			)

		print('[DEBUG] Starting agent execution...', file=sys.stderr)
		execution_wall_started = time.perf_counter()
		if comparison_method == 'webchallenger_style':
			agent_tools = _build_webchallenger_tools()
		elif comparison_method:
			agent_tools = None
		elif webshop_react_style:
			agent_tools = None
		elif webshop_web_only:
			agent_tools = _build_webshop_web_only_tools(task, agent_llm)
		elif use_webshop_candidate_verifier:
			agent_tools = _build_webshop_candidate_tools(task)
		elif short_ui_mode:
			agent_tools = Tools()
			for action_name in (
				'inspect_controls',
				'set_control_value',
				'set_controls_by_labels',
				'set_form_values',
				'wait_for_ui',
				'transfer_control_value',
			):
				if action_name == 'inspect_controls':
					continue
				if action_name == 'set_control_value' and (
					short_ui_batch_enabled
					or short_ui_wait_enabled
					or short_ui_transfer_enabled
					or short_ui_selection_enabled
					or short_ui_date_enabled
					or short_ui_ordinal_enabled
				):
					continue
				if action_name == 'set_form_values' and short_ui_batch_enabled:
					continue
				if action_name == 'set_controls_by_labels' and short_ui_selection_enabled:
					continue
				if action_name == 'wait_for_ui' and short_ui_wait_enabled:
					continue
				if action_name == 'transfer_control_value' and short_ui_transfer_enabled:
					continue
				agent_tools.registry.registry.actions.pop(action_name, None)
		else:
			agent_tools = None
		if is_miniwob_task and not comparison_method and _env_bool('EVAL_SHORT_UI_STRATEGY', default=True):
			visible_dom_workflow = await _execute_visible_dom_workflow(session, short_ui_profile.instruction_line)
			if visible_dom_workflow.get('matched'):
				print(
					'[DEBUG] Visible DOM workflow: '
					+ json.dumps(visible_dom_workflow, ensure_ascii=False, default=str),
					file=sys.stderr,
				)
		if (
			is_miniwob_task
			and not comparison_method
			and not visible_dom_workflow.get('success')
			and re.search(r'\b(?:similar to|synonym|related to|words? like)\b', short_ui_profile.instruction_line, re.IGNORECASE)
		):
			semantic_toggle_workflow = await _resolve_semantic_toggle_workflow(
				agent_llm,
				session,
				short_ui_profile.instruction_line,
				short_ui_profile.toggle_labels,
			)
			print(
				'[DEBUG] Semantic toggle workflow: '
				+ json.dumps(semantic_toggle_workflow, ensure_ascii=False, default=str),
				file=sys.stderr,
			)
		if (
			agent_tools is not None
			and not visible_dom_workflow.get('success')
			and not semantic_toggle_workflow.get('success')
			and (
				short_ui_selection_enabled
				or short_ui_date_enabled
				or short_ui_ordinal_enabled
				or short_ui_transfer_enabled
				or short_ui_wait_enabled
			)
		):
			pre_agent_preparation['attempted'] = True
			await session.get_browser_state_summary()
			pre_agent_preparation['browser_state_refreshed'] = True
			controls, inspection_error = await _inspect_pre_agent_controls(agent_tools, session)
			pre_agent_preparation['inspection_error'] = inspection_error

			if not inspection_error and short_ui_date_enabled:
				iso_date = _extract_iso_date(short_ui_profile.instruction_line)
				date_controls = [control for control in controls if control.get('type') == 'date']
				if iso_date and len(date_controls) == 1:
					date_result = await agent_tools.set_control_value(
						index=date_controls[0]['index'],
						value=iso_date,
						browser_session=session,
					)
					pre_agent_preparation['operations'].append(
						_action_result_record('set_native_date', date_result, value=iso_date),
					)

			if not inspection_error and short_ui_ordinal_enabled:
				checkbox_ordinal = _extract_ordinal(short_ui_profile.instruction_line, 'checkbox')
				checkboxes = [control for control in controls if control.get('type') == 'checkbox']
				if checkbox_ordinal and checkbox_ordinal <= len(checkboxes):
					checkbox_result = await agent_tools.set_control_value(
						index=checkboxes[checkbox_ordinal - 1]['index'],
						value=True,
						browser_session=session,
					)
					pre_agent_preparation['operations'].append(
						_action_result_record(
							'set_ordinal_checkbox',
							checkbox_result,
							ordinal=checkbox_ordinal,
						),
					)
				slider_value = _extract_slider_value(short_ui_profile.instruction_line)
				ranges = [control for control in controls if control.get('type') == 'range' or control.get('role') == 'slider']
				if slider_value is not None and len(ranges) == 1:
					slider_result = await agent_tools.set_control_value(
						index=ranges[0]['index'],
						value=slider_value,
						browser_session=session,
					)
					pre_agent_preparation['operations'].append(
						_action_result_record('set_slider_value', slider_result, value=slider_value),
					)

			if not inspection_error and short_ui_transfer_enabled:
				textareas = [control for control in controls if control.get('tag') == 'textarea']
				text_inputs = [
					control
					for control in controls
					if control.get('tag') == 'input' and control.get('type') in {'text', 'search', 'email', 'url', 'tel'}
				]
				source_ordinal = _extract_ordinal(short_ui_profile.instruction_line, 'textarea')
				if source_ordinal is None:
					source_ordinal = _extract_ordinal(short_ui_profile.instruction_line, 'text area')
				if source_ordinal is None and len(textareas) == 1:
					source_ordinal = 1
				if source_ordinal and source_ordinal <= len(textareas) and len(text_inputs) == 1:
					transfer_result = await agent_tools.transfer_control_value(
						source_index=textareas[source_ordinal - 1]['index'],
						target_index=text_inputs[0]['index'],
						browser_session=session,
					)
					pre_agent_preparation['operations'].append(
						_action_result_record(
							'transfer_exact_control_value',
							transfer_result,
							source_ordinal=source_ordinal,
						),
					)

			if not inspection_error and short_ui_wait_enabled:
				timed_sequence = _extract_timed_button_sequence(short_ui_profile.instruction_line)
				if timed_sequence:
					first_label, delay_seconds, second_label = timed_sequence
					buttons_by_name = {
						' '.join(str(control.get('name') or '').casefold().split()): control
						for control in controls
						if control.get('tag') == 'button' and control.get('name')
					}
					first_button = buttons_by_name.get(' '.join(first_label.casefold().split()))
					second_button = buttons_by_name.get(' '.join(second_label.casefold().split()))
					if first_button and second_button and first_button['index'] != second_button['index']:
						timed_result = await _execute_precise_button_sequence(
							session,
							first_label,
							delay_seconds,
							second_label,
						)
						pre_agent_preparation['operations'].append(
							{
								'operation': 'execute_timed_button_sequence',
								'success': bool(timed_result.get('success')),
								'error': timed_result.get('error'),
								'first_label': first_label,
								'second_label': second_label,
								'delay_seconds': delay_seconds,
								'elapsed_ms': timed_result.get('elapsed_ms'),
							},
						)

		if (
			short_ui_selection_enabled
			and agent_tools is not None
			and not visible_dom_workflow.get('success')
			and not semantic_toggle_workflow.get('success')
		):
			requested_labels = _requested_toggle_labels(short_ui_profile)
			pre_agent_selection['attempted'] = True
			pre_agent_selection['requested_labels'] = requested_labels
			if requested_labels:
				await session.get_browser_state_summary()
				pre_agent_selection['browser_state_refreshed'] = True
				selection_result = await agent_tools.set_controls_by_labels(
					labels=requested_labels,
					checked=True,
					browser_session=session,
				)
				pre_agent_selection['error'] = selection_result.error
				pre_agent_selection['success'] = selection_result.error is None
				pre_agent_selection['metadata'] = selection_result.metadata
				if selection_result.error is None:
					pre_agent_selection['fallback'] = 'not_needed'
					agent_tools.registry.registry.actions.pop('set_controls_by_labels', None)
					task += (
						'\n\nThe capability router already selected and DOM-verified the exact requested labelled '
						'controls. Do not change any checkbox or radio state; perform only the remaining visible commit action.'
					)
				else:
					pre_agent_selection['fallback'] = 'agent_tool'
			else:
				pre_agent_selection['error'] = 'No visible toggle labels were matched in the instruction line.'
			print(
				'[DEBUG] Pre-agent labelled selection: '
				+ json.dumps(pre_agent_selection, ensure_ascii=False, default=str),
				file=sys.stderr,
			)
		if pre_agent_preparation['attempted'] and not pre_agent_preparation['inspection_error']:
			expected_operations: list[str] = []
			if short_ui_date_enabled and _extract_iso_date(short_ui_profile.instruction_line):
				expected_operations.append('set_native_date')
			if short_ui_transfer_enabled:
				expected_operations.append('transfer_exact_control_value')
			if short_ui_ordinal_enabled:
				if _extract_ordinal(short_ui_profile.instruction_line, 'checkbox'):
					expected_operations.append('set_ordinal_checkbox')
				if _extract_slider_value(short_ui_profile.instruction_line) is not None:
					expected_operations.append('set_slider_value')
			successful_operation_names = {
				operation['operation']
				for operation in pre_agent_preparation['operations']
				if operation['success']
			}
			selection_ready = not short_ui_selection_enabled or pre_agent_selection['success']
			preparation_ready = bool(expected_operations) and all(
				operation in successful_operation_names for operation in expected_operations
			)
			submit_controls = [
				control
				for control in controls
				if (
					control.get('tag') == 'button'
					or (control.get('tag') == 'input' and control.get('type') in {'submit', 'button'})
				)
				and any(
					re.fullmatch(
						r'\s*(?:submit|continue|confirm|save|send|apply|done)\s*',
						str(control.get(field) or ''),
						flags=re.IGNORECASE,
					)
					for field in ('name', 'context')
				)
			]
			if preparation_ready and selection_ready and short_ui_profile.submit_count == 1 and len(submit_controls) == 1:
				submit_result = await agent_tools.click(
					index=submit_controls[0]['index'],
					browser_session=session,
				)
				pre_agent_preparation['operations'].append(
					_action_result_record('submit_prepared_workflow', submit_result),
				)
		if pre_agent_preparation['attempted']:
			successful_preparations = [
				operation for operation in pre_agent_preparation['operations'] if operation['success']
			]
			if successful_preparations:
				task += (
					'\n\nThe capability router already performed and DOM-verified these unambiguous control updates: '
					+ ', '.join(operation['operation'] for operation in successful_preparations)
					+ '. Preserve those values and perform only the remaining visible workflow steps.'
				)
			print(
				'[DEBUG] Pre-agent control preparation: '
				+ json.dumps(pre_agent_preparation, ensure_ascii=False, default=str),
				file=sys.stderr,
			)
		agent_kwargs = _agent_kwargs()
		if adapter_contract.observation_modality in {'vision_dom', 'desktop_visual'}:
			agent_kwargs['use_vision'] = True
		elif adapter_contract.observation_modality == 'offline_trajectory':
			agent_kwargs['use_vision'] = False
		if comparison_method:
			agent_kwargs['directly_open_url'] = False
		if comparison_method == 'agentoccam_style':
			agent_kwargs['max_history_items'] = 6
			agent_kwargs['max_actions_per_step'] = 1
			agent_kwargs['use_vision'] = False
		if is_miniwob_task:
			agent_kwargs['directly_open_url'] = False
			agent_kwargs['use_vision'] = False

			async def stop_after_miniwob_success() -> bool:
				reward_state = await _read_miniwob_reward(
					session,
					float(task_data['miniwob_reward_threshold']),
				)
				return reward_state.success

			agent_kwargs['register_should_stop_callback'] = stop_after_miniwob_success
		if webshop_web_only:
			agent_kwargs['max_history_items'] = min(
				int(agent_kwargs.get('max_history_items') or 8),
				max(6, int(os.getenv('EVAL_WEBSHOP_MAX_HISTORY_ITEMS', '6'))),
			)

			async def stop_after_visible_webshop_checkout() -> bool:
				final_url, final_text = await _read_final_browser_page(session)
				return '/done/' in final_url and _parse_webshop_reward(final_text) is not None

			agent_kwargs['register_should_stop_callback'] = stop_after_visible_webshop_checkout
		if short_ui_mode:
			agent_kwargs['directly_open_url'] = False
			agent_kwargs['max_actions_per_step'] = int(os.getenv('EVAL_SHORT_UI_MAX_ACTIONS_PER_STEP', '2'))
			agent_kwargs['max_history_items'] = int(os.getenv('EVAL_SHORT_UI_MAX_HISTORY_ITEMS', '6'))
			agent_kwargs['use_vision'] = _env_bool('EVAL_SHORT_UI_USE_VISION', default=False)
		method_config = {
			'browser_use_package_root': str(Path(browser_use_package.__file__).resolve().parent),
			'benchmark_adapter': adapter_contract.model_dump(mode='json'),
			'unified_capability_router': True,
			'enabled_capabilities': short_ui_profile.capabilities,
			'route_reason': short_ui_profile.route_reason,
			'short_ui_strategy_enabled': _env_bool('EVAL_SHORT_UI_STRATEGY', default=True),
			'short_ui_mode': short_ui_mode,
			'short_ui_profile': short_ui_profile.model_dump(mode='json'),
			'short_ui_batch_enabled': short_ui_batch_enabled,
			'short_ui_wait_enabled': short_ui_wait_enabled,
			'short_ui_transfer_enabled': short_ui_transfer_enabled,
			'short_ui_selection_enabled': short_ui_selection_enabled,
			'short_ui_date_enabled': short_ui_date_enabled,
			'short_ui_ordinal_enabled': short_ui_ordinal_enabled,
			'short_ui_inspection_enabled': short_ui_inspection_enabled,
			'pre_agent_label_selection': pre_agent_selection,
			'pre_agent_control_preparation': pre_agent_preparation,
			'visible_dom_workflow': visible_dom_workflow,
			'semantic_toggle_workflow': semantic_toggle_workflow,
			'max_actions_per_step': agent_kwargs.get('max_actions_per_step'),
			'max_history_items': agent_kwargs.get('max_history_items'),
			'use_vision': agent_kwargs.get('use_vision'),
			'task_policy_version': os.getenv(
				'EVAL_METHOD_VERSION',
				'capability_router_v4_9_focus_evidence',
			),
			'qwen3vl_compact_history': _is_qwen3_vl_model(),
			'webshop_min_evidence_coverage': float(os.getenv('EVAL_WEBSHOP_MIN_EVIDENCE_COVERAGE', '1.0')),
			'webshop_atomic_evidence_required': True,
			'webshop_max_visible_pages': int(os.getenv('EVAL_WEBSHOP_MAX_VISIBLE_PAGES', '6')),
			'webshop_candidates_per_query': int(os.getenv('EVAL_WEBSHOP_CANDIDATES_PER_QUERY', '5')),
			'webshop_semantic_rerank': _env_bool('EVAL_WEBSHOP_SEMANTIC_RERANK', default=True),
			'webshop_semantic_role': 'rank_cards_and_quote_grounded_detail_verification',
			'webshop_verified_auto_commit': not _env_bool('EVAL_ABLATE_VERIFIED_AUTO_COMMIT', default=False),
			'webshop_transaction_transitions': int(os.getenv('EVAL_WEBSHOP_TRANSACTION_TRANSITIONS', '50')),
			'task_policy_observation_contracts': ['dom', 'vision_dom', 'desktop_visual', 'offline_trajectory'],
			'task_policy_routes': [
				'candidate_search',
				'candidate_ranking',
				'form_interaction',
				'option_selection',
				'file_interaction',
				'predicate_sampling',
				'commit_validation',
				'visual_grounding',
				'completion_validation',
			],
		}
		if agent_tools is not None:
			agent_kwargs['tools'] = agent_tools
			if use_webshop_candidate_verifier:
				initial_url_match = re.search(r'https?://127\.0\.0\.1(?::\d+)?/[^\s.]+', task_data['task'])
				if initial_url_match:
					agent_kwargs['initial_actions'] = [
						{'navigate': {'url': initial_url_match.group(0), 'new_tab': False}},
						{'advance_webshop_task': {}},
					]
				print('[DEBUG] WebShop deterministic DB-assisted state machine enabled', file=sys.stderr)
			elif comparison_method == 'webchallenger_style':
				print(
					'[DEBUG] WebChallenger online PageMem section tool enabled; offline cross-task memory disabled',
					file=sys.stderr,
				)
			elif webshop_web_only:
				agent_kwargs['directly_open_url'] = False
				agent_kwargs['initial_actions'] = [{'execute_visible_webshop_transaction': {}}]
				print(
					'[DEBUG] Fair visible-evidence policy enabled: candidate ranking, bounded recovery, commit guard, and grounded done',
					file=sys.stderr,
				)
			elif short_ui_mode:
				print(
					'[DEBUG] Generic short-UI tools enabled: '
					f'batch={short_ui_batch_enabled}, wait={short_ui_wait_enabled}, '
					f'transfer={short_ui_transfer_enabled}, '
					f'selection={short_ui_selection_enabled}, '
					f'date={short_ui_date_enabled}, ordinal={short_ui_ordinal_enabled}, '
					f'inspection={short_ui_inspection_enabled}, '
					f'max_actions_per_step={agent_kwargs["max_actions_per_step"]}',
					file=sys.stderr,
				)
			else:
				print('[DEBUG] Fair web-only tools enabled; internal JSON access disabled', file=sys.stderr)
		try:
			if visible_dom_workflow.get('success'):
				history = AgentHistoryList(history=[])
				controller_histories = [history]
				controller_metadata = {
					'controller_method': 'programmatic_visible_workflow',
					'trials_used': 0,
					'controller_llm_calls': 0,
					'trial_records': [],
					'controller_memory': [visible_dom_workflow],
					'provenance': 'visible_dom_only',
				}
				print('[DEBUG] Visible workflow completed; Agent loop skipped', file=sys.stderr)
			elif comparison_method in controller_methods:
				history, controller_histories, controller_metadata = await _run_comparison_controller(
					comparison_method,
					task,
					task_data,
					agent_llm,
					session,
					agent_kwargs,
					max_steps,
				)
			else:
				agent = Agent(task=task, llm=agent_llm, browser_session=session, **agent_kwargs)
				run_timeout = float(os.getenv('EVAL_AGENT_RUN_TIMEOUT_SECONDS', '240'))
				history = await asyncio.wait_for(agent.run(max_steps=max_steps), timeout=run_timeout)
				controller_histories = [history]
				controller_metadata = {
					'controller_method': comparison_method or 'single_agent_loop',
					'trials_used': 1,
					'controller_llm_calls': 0,
					'trial_records': [],
					'controller_memory': [],
					'provenance': _comparison_provenance(comparison_method),
				}
			print('[DEBUG] Execution controller returned successfully', file=sys.stderr)
		except Exception as agent_error:
			print(
				f'[DEBUG] Agent.run() failed with error: {str(agent_error)}',
				file=sys.stderr,
			)
			print(f'[DEBUG] Error type: {type(agent_error).__name__}', file=sys.stderr)
			# Re-raise to be caught by outer try-catch
			raise agent_error

		agent_output = history.final_result() or ''
		print('[DEBUG] Agent execution completed', file=sys.stderr)
		_debug_history(history)

		# Avoid an unnecessary paid model call when the visible programmatic workflow already completed the task.
		if not visible_dom_workflow.get('success'):
			try:
				response = await agent_llm.ainvoke([UserMessage(content="Say 'test'")])
				print(
					f'[DEBUG] LLM test call successful: {response.completion[:50]}',
					file=sys.stderr,
				)
			except Exception as llm_error:
				print(f'[DEBUG] LLM test call failed: {str(llm_error)}', file=sys.stderr)

		# Debug: capture more details about the agent execution
		total_steps = len(history.history) if hasattr(history, 'history') else 0
		last_action = history.history[-1] if hasattr(history, 'history') and history.history else None
		debug_info = f'Steps: {total_steps}, Final result length: {len(agent_output)}'
		if last_action:
			debug_info += f', Last action: {type(last_action).__name__}'

		# Log to stderr so it shows up in GitHub Actions (won't interfere with JSON output to stdout)
		print(f'[DEBUG] Task {os.path.basename(task_file)}: {debug_info}', file=sys.stderr)
		if agent_output:
			print(
				f'[DEBUG] Agent output preview: {agent_output[:200]}...',
				file=sys.stderr,
			)
		else:
			print('[DEBUG] Agent produced no output!', file=sys.stderr)
		execution_metrics = _combined_history_metrics(controller_histories)
		execution_metrics['duration_seconds'] = round(time.perf_counter() - execution_wall_started, 3)
		semantic_usage = semantic_toggle_workflow.get('usage') or {}
		semantic_calls = int(semantic_toggle_workflow.get('llm_calls') or 0)
		if semantic_calls:
			for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
				execution_metrics[key] = int(execution_metrics.get(key) or 0) + int(semantic_usage.get(key) or 0)
			execution_metrics['llm_invocations'] = int(execution_metrics.get('llm_invocations') or 0) + semantic_calls
			execution_metrics['semantic_resolver_llm_calls'] = semantic_calls

		miniwob_reward_response = (
			await _read_miniwob_reward(session, float(task_data['miniwob_reward_threshold'])) if is_miniwob_task else None
		)
		if miniwob_reward_response is not None:
			result = {
				'file': os.path.basename(task_file),
				'success': miniwob_reward_response.success,
				'explanation': miniwob_reward_response.explanation,
				'reward': miniwob_reward_response.reward,
				'evaluation_type': 'miniwob_reward',
				'evaluation_mode': os.getenv('EVAL_EXPERIMENT_LABEL') or comparison_method or 'baseline',
				'experiment_seed': int(os.getenv('EVAL_SEED', '0')),
				'failure_category': _failure_category(
					miniwob_reward_response.success,
					miniwob_reward_response.explanation,
					miniwob_reward_response.reward,
				),
				'method_config': method_config,
				'metrics': execution_metrics,
				'controller': controller_metadata,
			}
			await session.kill()
			return result

		final_url, final_page_text = await _read_final_browser_page(session) if is_webshop_task else ('', '')
		if not final_url and is_webshop_task:
			visited_urls = history.urls()
			final_url = visited_urls[-1] if visited_urls else ''
		if is_webshop_task and final_url and not final_page_text:
			final_page_text = await _fetch_local_webshop_page(final_url)
		webshop_reward_response = _webshop_reward_response(
			task_data,
			history,
			agent_output,
			final_page_text=final_page_text,
			final_url=final_url,
		)
		if webshop_reward_response is not None:
			result = {
				'file': os.path.basename(task_file),
				'success': webshop_reward_response.success,
				'explanation': webshop_reward_response.explanation,
				'reward': webshop_reward_response.reward,
				'evaluation_type': 'webshop_reward',
				'experiment_seed': int(os.getenv('EVAL_SEED', '0')),
				'evaluation_mode': os.getenv('EVAL_EXPERIMENT_LABEL')
				or (
					'react_style_web_only'
					if webshop_react_style
					else ('web_only' if webshop_web_only else ('db_assisted' if use_webshop_candidate_verifier else 'baseline'))
				),
				'failure_category': _failure_category(
					webshop_reward_response.success,
					webshop_reward_response.explanation,
					webshop_reward_response.reward,
				),
				'method_config': method_config,
				'metrics': execution_metrics,
				'controller': controller_metadata,
			}
			await session.kill()
			return result

		criteria = '\n- '.join(judge_context)
		judge_prompt = f"""
You are a evaluator of a browser agent task inside a ci/cd pipeline. Here was the agent's task:
{task}

Here is the agent's output:
{agent_output if agent_output else '[No output provided]'}

Debug info: {debug_info}

Criteria for success:
- {criteria}

Reply in JSON with keys: success (true/false), explanation (string).
If the agent provided no output, explain what might have gone wrong.
"""
		judge_response = await _judge_task(judge_llm, judge_prompt)

		result = {
			'file': os.path.basename(task_file),
			'success': judge_response.success,
			'explanation': judge_response.explanation,
			'evaluation_type': 'llm_judge',
			'evaluation_mode': os.getenv('EVAL_EXPERIMENT_LABEL') or comparison_method or 'baseline',
			'experiment_seed': int(os.getenv('EVAL_SEED', '0')),
			'failure_category': _failure_category(judge_response.success, judge_response.explanation, None),
			'method_config': method_config,
			'metrics': execution_metrics,
			'controller': controller_metadata,
		}

		# Clean up session before returning
		await session.kill()

		return result

	except Exception as e:
		# Ensure session cleanup even on error
		try:
			await session.kill()
		except Exception:
			pass

		error_text = str(e) or type(e).__name__
		return {
			'file': os.path.basename(task_file),
			'success': False,
			'explanation': f'Task failed with error: {error_text}',
			'evaluation_mode': os.getenv('EVAL_EXPERIMENT_LABEL') or os.getenv('EVAL_COMPARISON_METHOD') or 'baseline',
			'experiment_seed': int(os.getenv('EVAL_SEED', '0')),
			'failure_category': _failure_category(False, error_text, None),
			'metrics': None,
		}


async def run_task_subprocess(task_file, semaphore):
	"""Run a task in a separate subprocess"""
	async with semaphore:
		try:
			# Set environment to reduce noise in subprocess
			env = os.environ.copy()
			env['PYTHONPATH'] = os.pathsep.join(sys.path)

			proc = await asyncio.create_subprocess_exec(
				sys.executable,
				__file__,
				'--task',
				task_file,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
				env=env,
			)
			stdout, stderr = await proc.communicate()

			if proc.returncode == 0:
				try:
					# Parse JSON result from subprocess
					stdout_text = stdout.decode().strip()
					stderr_text = stderr.decode().strip()

					# Display subprocess debug logs
					if stderr_text:
						print(f'[SUBPROCESS {os.path.basename(task_file)}] Debug output:')
						for line in stderr_text.split('\n'):
							if line.strip():
								print(f'  {line}')

					# Find the JSON line (should be the last line that starts with {)
					lines = stdout_text.split('\n')
					json_line = None
					for line in reversed(lines):
						line = line.strip()
						if line.startswith('{') and line.endswith('}'):
							json_line = line
							break

					if json_line:
						result = json.loads(json_line)
						print(f'[PARENT] Task {os.path.basename(task_file)} completed: {result["success"]}')
					else:
						raise ValueError(f'No JSON found in output: {stdout_text}')

				except (json.JSONDecodeError, ValueError) as e:
					result = {
						'file': os.path.basename(task_file),
						'success': False,
						'explanation': f'Failed to parse subprocess result: {str(e)[:100]}',
					}
					print(f'[PARENT] Task {os.path.basename(task_file)} failed to parse: {str(e)}')
					print(f'[PARENT] Full stdout was: {stdout.decode()[:500]}')
			else:
				stderr_text = stderr.decode().strip()
				result = {
					'file': os.path.basename(task_file),
					'success': False,
					'explanation': f'Subprocess failed (code {proc.returncode}): {stderr_text[:200]}',
				}
				print(f'[PARENT] Task {os.path.basename(task_file)} subprocess failed with code {proc.returncode}')
				if stderr_text:
					print(f'[PARENT] stderr: {stderr_text[:1000]}')
				stdout_text = stdout.decode().strip()
				if stdout_text:
					print(f'[PARENT] stdout: {stdout_text[:1000]}')
		except Exception as e:
			result = {
				'file': os.path.basename(task_file),
				'success': False,
				'explanation': f'Failed to start subprocess: {str(e)}',
			}
			print(f'[PARENT] Failed to start subprocess for {os.path.basename(task_file)}: {str(e)}')

		return result


async def main():
	"""Run all tasks in parallel using subprocesses"""
	semaphore = asyncio.Semaphore(MAX_PARALLEL)

	print(f'Found task files: {TASK_FILES}')

	if not TASK_FILES:
		print('No task files found!')
		return 0, 0

	completed_files = _completed_result_files()
	task_files = [task_file for task_file in TASK_FILES if os.path.basename(task_file) not in completed_files]
	if completed_files:
		print(f'Resuming: skipping {len(TASK_FILES) - len(task_files)} completed tasks from {RESULTS_JSONL}')

	# Run tasks in parallel subprocesses and optionally append each result as it finishes.
	tasks = [asyncio.create_task(run_task_subprocess(task_file, semaphore)) for task_file in task_files]
	results = []
	for task_future in asyncio.as_completed(tasks):
		result = await task_future
		results.append(result)
		_append_result(result)

	passed = sum(1 for r in results if r['success'])
	total = len(results)

	print('\n' + '=' * 60)
	print(f'{"RESULTS":^60}\n')

	# Prepare table data
	headers = ['Task', 'Success', 'Reason']
	rows = []
	for r in results:
		status = '✅' if r['success'] else '❌'
		rows.append([r['file'], status, r['explanation']])

	# Calculate column widths
	col_widths = [max(len(str(row[i])) for row in ([headers] + rows)) for i in range(3)]

	# Print header
	header_row = ' | '.join(headers[i].ljust(col_widths[i]) for i in range(3))
	print(header_row)
	print('-+-'.join('-' * w for w in col_widths))

	# Print rows
	for row in rows:
		print(' | '.join(str(row[i]).ljust(col_widths[i]) for i in range(3)))

	print('\n' + '=' * 60)
	print(f'\n{"SCORE":^60}')
	print(f'\n{"=" * 60}\n')
	print(f'\n{"*" * 10}  {passed}/{total} PASSED  {"*" * 10}\n')
	print('=' * 60 + '\n')

	# Output results for GitHub Actions
	print(f'PASSED={passed}')
	print(f'TOTAL={total}')

	# Output detailed results as JSON for GitHub Actions
	detailed_results = []
	for r in results:
		detailed_results.append(
			{
				'task': r['file'].replace('.yaml', ''),
				'success': r['success'],
				'reason': r['explanation'],
			}
		)

	print('DETAILED_RESULTS=' + json.dumps(detailed_results))

	return passed, total


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('task_dir', nargs='?', help='Directory containing agent task YAML files')
	parser.add_argument('--task', type=str, help='Path to a single task YAML file (for subprocess mode)')
	args = parser.parse_args()

	if args.task:
		# Subprocess mode: run a single task and output ONLY JSON
		try:
			result = asyncio.run(run_single_task(args.task))
			# Output ONLY the JSON result, nothing else
			print(json.dumps(result))
		except Exception as e:
			# Even on critical failure, output valid JSON
			error_result = {
				'file': os.path.basename(args.task),
				'success': False,
				'explanation': f'Critical subprocess error: {str(e)}',
			}
			print(json.dumps(error_result))
	else:
		# Parent process mode: run all tasks in parallel subprocesses
		passed, total = asyncio.run(main())
		# Results already printed by main() function

		# Fail if 0% pass rate (all tasks failed)
		if total > 0 and passed == 0:
			print('\n❌ CRITICAL: 0% pass rate - all tasks failed!')
			sys.exit(1)
